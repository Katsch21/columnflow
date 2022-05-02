from __future__ import print_function, division
from coffea.processor import Runner, ProcessorABC
from coffea.processor.executor import WorkItem, ParquetFileContext
from coffea.processor.dataframe import LazyDataFrame
from coffea.nanoevents import schemas, NanoEventsFactory
import uuid
import uproot
import time
import cloudpickle
import lz4.frame as lz4f
from dataclasses import asdict
from typing import Dict
import json
from collections.abc import MutableMapping
from typing import Callable

__all__ = ["RunnerWithPassThrough"]


class RunnerWithPassThrough(Runner):

    def __init__(self, *args, **kwargs):
        print("initializing Runner with pass through")
        self.__processor_passthrough = kwargs.get("processor_passthrough", {})
        print("pass through arguments:")
        print(json.dumps(self.__processor_passthrough, indent=4))
        kwargs.pop("processor_passthrough", None)
        super().__init__(*args, **kwargs)
        print("done initializing!")
        print(type(self))

    @staticmethod
    def _work_function(
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
                    factory = NanoEventsFactory.from_root(
                        file=file,
                        treepath=item.treename,
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                        persistent_cache=cache_function(),
                        schemaclass=schema,
                        metadata=metadata,
                        access_log=materialized,
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

                    factory = NanoEventsFactory.from_parquet(
                        file=item.filename,
                        treepath=item.treename,
                        schemaclass=schema,
                        metadata=metadata,
                        skyhook_options=skyhook_options,
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

    # def __call__(
    #     self,
    #     fileset: Dict,
    #     treename: str,
    #     processor_instance: ProcessorABC,
    # ) -> Accumulatable:
    #     """Run the processor_instance on a given fileset
    #     Parameters
    #     ----------
    #         fileset : dict
    #             A dictionary ``{dataset: [file, file], }``
    #             Optionally, if some files' tree name differ, the dictionary can be specified:
    #             ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
    #         treename : str
    #             name of tree inside each root file, can be ``None``;
    #             treename can also be defined in fileset, which will override the passed treename
    #         processor_instance : ProcessorABC
    #             An instance of a class deriving from ProcessorABC
    #     """

    #     if not isinstance(fileset, (Mapping, str)):
    #         raise ValueError(
    #             "Expected fileset to be a mapping dataset: list(files) or filename"
    #         )
    #     if not isinstance(processor_instance, ProcessorABC):
    #         raise ValueError("Expected processor_instance to derive from ProcessorABC")

    #     if self.format == "root":
    #         fileset = list(self._normalize_fileset(fileset, treename))
    #         for filemeta in fileset:
    #             filemeta.maybe_populate(self.metadata_cache)

    #         self._preprocess_fileset(fileset)
    #         fileset = self._filter_badfiles(fileset)

    #         # reverse fileset list to match the order of files as presented in version
    #         # v0.7.4. This fixes tests using maxchunks.
    #         fileset.reverse()

    #     chunks = self._chunk_generator(fileset, treename)

    #     if self.processor_compression is None:
    #         pi_to_send = processor_instance
    #     else:
    #         pi_to_send = lz4f.compress(
    #             cloudpickle.dumps(processor_instance),
    #             compression_level=self.processor_compression,
    #         )
    #     # hack around dask/dask#5503 which is really a silly request but here we are
    #     if isinstance(self.executor, DaskExecutor):
    #         self.executor.heavy_input = pi_to_send
    #         closure = partial(self._work_function, processor_instance="heavy")
    #     else:
    #         closure = partial(self._work_function, processor_instance=pi_to_send,
    #                         passthrough_to_processor=self.__processor_passthrough)

    #     if self.format == "root":
    #         if self.dynamic_chunksize:
    #             events_total = sum(f.metadata["numentries"] for f in fileset)
    #         else:
    #             chunks = [c for c in chunks]
    #             events_total = sum(len(c) for c in chunks)
    #     else:
    #         chunks = [c for c in chunks]

    #     exe_args = {
    #         "unit": "event" if isinstance(self.executor, WorkQueueExecutor) else "chunk",  # fmt: skip
    #         "function_name": type(processor_instance).__name__,
    #     }
    #     if self.format == "root" and isinstance(self.executor, WorkQueueExecutor):
    #         exe_args.update(
    #             {
    #                 "events_total": events_total,
    #                 "dynamic_chunksize": self.dynamic_chunksize,
    #                 "chunksize": self.chunksize,
    #             }
    #         )

    #     closure = partial(self.automatic_retries, closure)

    #     executor = self.executor.copy(**exe_args)
    #     wrapped_out = executor(chunks, closure, None)

    #     processor_instance.postprocess(wrapped_out["out"])
    #     if self.savemetrics and not self.use_dataframes:
    #         wrapped_out["metrics"]["chunks"] = len(chunks)
    #         return wrapped_out["out"], wrapped_out["metrics"]
    #     return wrapped_out["out"]
