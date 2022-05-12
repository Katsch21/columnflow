from __future__ import print_function, division
from coffea.processor import Runner, ProcessorABC
from coffea.processor.executor import WorkItem, ParquetFileContext
from coffea.processor.dataframe import LazyDataFrame
from coffea.nanoevents import schemas, NanoEventsFactory
import uuid
import uproot
import time
import cloudpickle
import awkward
import weakref
import lz4.frame as lz4f
from dataclasses import asdict
from typing import Dict
from collections.abc import MutableMapping
from typing import Callable
from functools import partial

__all__ = ["RunnerWithPassThrough"]


def _key_formatter(prefix, partition, form_key, attribute):
    return prefix + f"/{partition}/{form_key}/{attribute}"


class EagerNanoEventsFactory(NanoEventsFactory):
    """
    NanoEventsFactory which includes a switch for lazy/greedy IO of events.
    """

    def events(self):
        """Build events"""
        print("initializing eager events")
        events = self._events()
        if events is None:
            behavior = dict(self._schema.behavior)
            behavior["__events_factory__"] = self
            events = awkward.from_buffers(
                self._schema.form,
                len(self),
                self._mapping,
                key_format=partial(_key_formatter, self._partition_key),
                behavior=behavior,
            )
            self._events = weakref.ref(events)

        return events


class RunnerWithPassThrough(Runner):

    def __init__(self, *args, **kwargs):
        """
        Init function of sub class. This function extends the init of the 
        *Runner* class with the following keywords:

        processor_passthrough (dict): commands that need to be passed to the
                                      processor instance (right now does nothing)
        read_options (dict):            commands that are passed to uproot
                                        or parquet via the NanoEventsFactory
        """
        # collect the extensions from the keyword arguments
        self.__processor_passthrough = kwargs.get("processor_passthrough", {})
        kwargs.pop("processor_passthrough", None)
        self.__read_options = kwargs.get("read_options", {})
        kwargs.pop("read_options", None)
        self.__io_mode = kwargs.get("io_mode", "lazy")
        kwargs.pop("io_mode", None)

        # pass all other arguments on to init function of super
        super().__init__(*args, **kwargs)

    @property
    def read_options(self) -> Dict:
        return self.__read_options

    @read_options.setter
    def read_options(self, input_dict) -> None:
        if isinstance(input_dict, dict):
            self.__read_options = input_dict
        else:
            msg = f"property 'read_options' is a dict, received {type(input_dict)}"
            raise ValueError(msg)

    @property
    def io_mode(self) -> str:
        return self.__io_mode

    @io_mode.setter
    def io_mode(self, value) -> None:
        if value not in ["lazy", "eager"]:
            raise ValueError("io_mode for events must by 'lazy' or 'eager'!")
        self.__io_mode = value

    # @staticmethod
    def _work_function(
        self,
        format: str,
        xrootdtimeout: int,
        mmap: bool,
        schema: schemas.BaseSchema,
        cache_function: Callable[[], MutableMapping],
        use_dataframes: bool,
        savemetrics: bool,
        item: WorkItem,
        processor_instance: ProcessorABC,
    ) -> Dict:
        if processor_instance == "heavy":
            item, processor_instance = item
        if not isinstance(processor_instance, ProcessorABC):
            processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

        if format == "root":
            filecontext = uproot.open(
                item.filename,
                timeout=xrootdtimeout,
                file_handler=uproot.MemmapSource
                if mmap
                else uproot.MultithreadedFileSource,
            )
        elif format == "parquet":
            filecontext = ParquetFileContext(item.filename)

        metadata = {
            "dataset": item.dataset,
            "filename": item.filename,
            "treename": item.treename,
            "entrystart": item.entrystart,
            "entrystop": item.entrystop,
            "fileuuid": str(uuid.UUID(bytes=item.fileuuid))
            if len(item.fileuuid) > 0
            else "",
        }
        if item.usermeta is not None:
            metadata.update(item.usermeta)

        event_factory = NanoEventsFactory if self.io_mode == "lazy" else EagerNanoEventsFactory
        with filecontext as file:
            if schema is None:
                # To deprecate
                tree = file[item.treename]
                events = LazyDataFrame(
                    tree, item.entrystart, item.entrystop, metadata=metadata
                )
            elif issubclass(schema, schemas.BaseSchema):
                # change here
                if format == "root":
                    materialized = []
                    factory = event_factory.from_root(
                        file=file,
                        treepath=item.treename,
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                        persistent_cache=cache_function(),
                        schemaclass=schema,
                        metadata=metadata,
                        access_log=materialized,
                        **self.__read_options
                    )
                    events = factory.events()
                elif format == "parquet":
                    skyhook_options = {}
                    if ":" in item.filename:
                        (
                            ceph_config_path,
                            ceph_data_pool,
                            filename,
                        ) = item.filename.split(":")
                        # patch back filename into item
                        item = WorkItem(**dict(asdict(item), filename=filename))
                        skyhook_options["ceph_config_path"] = ceph_config_path
                        skyhook_options["ceph_data_pool"] = ceph_data_pool

                    factory = event_factory.from_parquet(
                        file=item.filename,
                        treepath=item.treename,
                        schemaclass=schema,
                        metadata=metadata,
                        skyhook_options=skyhook_options,
                        **self.__read_options
                    )
                    events = factory.events()
            else:
                raise ValueError(
                    "Expected schema to derive from nanoevents.BaseSchema, instead got %r"
                    % schema
                )
            tic = time.time()
            try:
                if getattr(processor_instance, "request_metadata", False):
                    out = processor_instance.process(events, **metadata)
                else:
                    out = processor_instance.process(events)
            except Exception as e:
                raise Exception(f"Failed processing file: {item!r}") from e
            if out is None:
                raise ValueError(
                    "Output of process() should not be None. Make sure your processor's process() function returns an accumulator."
                )
            toc = time.time()
            if use_dataframes:
                return out
            else:
                if savemetrics:
                    metrics = {}
                    if isinstance(file, uproot.ReadOnlyDirectory):
                        metrics["bytesread"] = file.file.source.num_requested_bytes
                    if schema is not None and issubclass(schema, schemas.BaseSchema):
                        metrics["columns"] = set(materialized)
                        metrics["entries"] = len(events)
                    else:
                        metrics["columns"] = set(events.materialized)
                        metrics["entries"] = events.size
                    metrics["processtime"] = toc - tic
                    return {"out": out, "metrics": metrics, "processed": set([item])}
                return {"out": out, "processed": set([item])}
