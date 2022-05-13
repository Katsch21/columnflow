# coding: utf-8

"""
Tasks related to obtaining, preprocessing and selecting events.
"""

import math
from secrets import choice

import law
import luigi
import awkward as ak
import os

from ap.tasks.framework import DatasetTask, HTCondorWorkflow
from ap.util import ensure_proxy

from coffea.processor import ProcessorABC, list_accumulator
from coffea import processor, nanoevents
from ap.coffea_util import RunnerWithPassThrough
import concurrent.futures as cf


class CalibrateObjects(DatasetTask, law.LocalWorkflow, HTCondorWorkflow):

    sandbox = "bash::$AP_BASE/sandboxes/venv_selection.sh"
    nworkers = luigi.IntParameter(
        default=4,
        description="number of workers to be used for multiprocessing; default: 4",
    )
    io_mode = luigi.ChoiceParameter(
        choices=["lazy", "eager"],
        default="lazy",
        description="mode for io of events. Choices: lazy, eager",
    )

    def workflow_requires(self):
        # workflow super classes might already define requirements, so extend them
        reqs = super(CalibrateObjects, self).workflow_requires()
        reqs["lfns"] = GetDatasetLFNs.req(self)
        return reqs

    def requires(self):
        # workflow branches are normal tasks, so define requirements the normal way
        return {"lfns": GetDatasetLFNs.req(self)}

    def output(self):
        return self.local_target(f"data_{self.branch}.parquet")

    @law.decorator.safe_output
    @ensure_proxy
    def run(self):
        # get all lfns
        lfns = self.input()["lfns"].random_target().load(formatter="json")

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()
        # output_chunks = []

        # loop over all input file indices requested by this branch
        for file_index in self.branch_data:
            self.publish_message(f"handling file {file_index}")

            # get the lfn of the file referenced by this file index
            lfn = str(lfns[file_index])

            # always use the INFN redirector for now
            input_file = law.wlcg.WLCGFileTarget(lfn, fs="wlcg_fs_infn")
            # input_file = law.law.LocalFileTarget("/nfs/dust/cms/user/pkeicher/uhh/u4h/analysis_playground/local_testbox/2325C825-4095-D74F-A98A-5B42318F8DC4.root")
            input_size = law.util.human_bytes(input_file.stat().st_size, fmt=True)
            self.publish_message(f"lfn {lfn}, size is {input_size}")

            class ParquetProcessor(ProcessorABC):

                def __init__(self) -> None:
                    super().__init__()
                    self._accumulator = list_accumulator()
                    self.request_metadata = True
                    self.output_fname = "local_testbox/test.parquet"
                    self.__law_task = None
                    self.prefix = "branch_"
                @property
                def accumulator(self):
                    return self._accumulator

                @property
                def law_task(self):
                    return self.__law_task

                @law_task.setter
                def law_task(self, task):
                    self.__law_task = task

                # def process(self, events, entry_start, entry_stop, outdir):
                def process(self, events, **kwargs):
                    # do stuff with the events
                    import os
                    import numpy as np
                    # "correct" Jet.pt by scaling four momenta by 1.1 (pt<30) or 0.9 (pt<=30)

                    entry_stop = kwargs.get("entrystop", -1)
                    entry_start = kwargs.get("entrystart", -1)
                    print(f"calibrating objects {entry_start} - {entry_stop}")

                    outdir = os.path.dirname(self.output_fname)
                    a_mask = ak.flatten(events.Jet.pt < 30)
                    n_jet_pt = np.asarray(ak.flatten(events.Jet.pt))
                    n_jet_mass = np.asarray(ak.flatten(events.Jet.mass))
                    n_jet_pt[a_mask] *= 1.1
                    n_jet_pt[~a_mask] *= 0.9
                    n_jet_mass[a_mask] *= 1.1
                    n_jet_mass[~a_mask] *= 0.9

                    events["Jet", "pt_jec_up"] = events.Jet.pt * 1.05
                    events["Jet", "mass_jec_up"] = events.Jet.mass * 1.05
                    events["Jet", "pt_jec_down"] = events.Jet.pt * 0.95
                    events["Jet", "mass_jec_down"] = events.Jet.mass * 0.95
                    
                    fname = os.path.join(outdir,
                                f"{self.prefix}_{entry_start}_{entry_stop}.parquet")
                    ak.to_parquet(events, fname)
                    self.accumulator.add([fname])
                    print("\texit")
                    return self.accumulator

                def postprocess(self, accumulator):
                    import law
                    # final_merge_path = "test.parquet"
                    # law.pyarrow.merge_parquet_files(sorted(accumulator), self.output_fname)
                    print("merging files")
                    sorted_accu = sorted(accumulator)
                    print(sorted_accu)
                    law.pyarrow.merge_parquet_task(self.law_task, sorted_accu,
                                            self.output_fname, force=True,
                                            local=True)
                    return accumulator

            load_columns = ["run", "luminosityBlock", "event"]
            load_columns += "Jet_pt Jet_mass nJet".split()
            # load_columns = ["*"]
            runner = RunnerWithPassThrough(
                executor=processor.FuturesExecutor(workers=self.nworkers,
                                                # pool=cf.ThreadPoolExecutor
                                                ),
                schema=nanoevents.NanoAODSchema,
                chunksize=40e3,
                read_options={"iteritems_options": {"filter_name": load_columns}},
            )
            runner.io_mode = self.io_mode

            proc = ParquetProcessor()
            proc.law_task = self
            proc.prefix = f"file_{file_index}"
            proc.output_fname = os.path.join(tmp_dir.path, f"file_{file_index}_corrected.parquet")
            with input_file.localize("r") as tmp:
                output = runner(
                    fileset={"Test": [tmp.path]},
                    treename="Events",
                    processor_instance=proc,
                )

            print(f"Created merged file in '{output}'")
            # open with uproot
            # with self.publish_step("load and open ..."):
            #     tree = input_file.load(formatter="uproot")["Events"]
            #     self.publish_message(f"found {tree.num_entries} events")

            # # prepare processing
            # chunk_size = 40000
            # n_chunks = int(math.ceil(tree.num_entries / chunk_size))
            # branches = ["run", "luminosityBlock", "event", "nJet", "Jet_*"]
            # branches = [k for k in tree.keys() if law.util.multi_match(k, branches)]

            # # iterate over the file
            # gen = process_nano_events(tree, chunk_size=chunk_size, expressions=branches)
            # for i, events, *_ in self.iter_progress(gen, n_chunks, msg="iterate ..."):
            #     # do sth meaningful here ...
            #     print(i, events.Jet.pt)

            # TODO 1: it looks like the NanoEventsArray copied the preloaded input chunk,
            #         so a) is this true?, b) is this normal?, and c) would this also be the
            #         case if we used "from_root" instead of "from_preloaded"?

            # TODO 2: how to extract a zipped awkward array again from the NanoEventsArray
            #         without copying?

            # TODO 3: how to save to parquet (which will again be I/O bound) using threads such
            #         that the next event processing iteration can start without the need to
            #         wait until the file is written

            # save the batch as parquet
            # chunk = tmp_dir.child(f"file_{file_index}_{i}.parquet", type="f")
            # ak.to_parquet(batch, chunk.path)
            # output_chunks.append(chunk)

        # merge the files
        # with self.output().localize("w") as output:
        #     law.pyarrow.merge_parquet_task(self, output_chunks, output, local=True)


class SelectEvents(DatasetTask, law.LocalWorkflow, HTCondorWorkflow):

    sandbox = "bash::$AP_BASE/sandboxes/venv_selection.sh"

    shifts = {"jec_up", "jec_down"}

    def workflow_requires(self):
        # workflow super classes might already define requirements, so extend them
        reqs = super(SelectEvents, self).workflow_requires()
        reqs["lfns"] = GetDatasetLFNs.req(self)
        return reqs

    def requires(self):
        # workflow branches are normal tasks, so define requirements the normal way
        return {"lfns": GetDatasetLFNs.req(self)}

    def output(self):
        return self.local_target(f"data_{self.branch}.npz")

    @law.decorator.safe_output
    @ensure_proxy
    def run(self):
        import numpy as np

        # get all lfns
        lfns = self.input()["lfns"].random_target().load(formatter="json")

        # prepare output arrays to be concated
        output_arrays = []

        # loop over all input file indices requested by this branch
        for file_index in self.branch_data:
            # get the lfn of the file referenced by this file index
            lfn = str(lfns[file_index])
            self.publish_message(f"file {file_index}: fround {lfn}")

            # always use the INFN redirector for now
            input_file = law.wlcg.WLCGFileTarget(lfn, fs="wlcg_fs_infn")

            # open with uproot
            with self.publish_step("loading content with uproot ..."):
                data = input_file.load(formatter="uproot")
                events = data["Events"]
                self.publish_message(f"file {file_index}: found {events.num_entries} events")

            # dummy task: get all jet 1 pt values
            step_size = 1000
            steps = int(math.ceil(events.num_entries / step_size))
            events = events.iterate(["nJet", "Jet_pt"], step_size=step_size)
            for batch in self.iter_progress(events, steps, msg=f"file {file_index}: select ..."):
                print("batch")
                mask = batch["nJet"] >= 1
                jet_pt = batch["Jet_pt"][mask][:, 0]

                # emulate jec
                jec_factor = {"jec_up": 1.05, "jec_down": 0.95}.get(self.effective_shift, 1.0)
                jet_pt = jet_pt * jec_factor

                # store the jet pt
                output_arrays.append(jet_pt.to_numpy())

        data = np.concatenate(output_arrays, axis=0)
        self.output().dump(data=data, formatter="numpy")


# trailing imports
from ap.tasks.external import GetDatasetLFNs
