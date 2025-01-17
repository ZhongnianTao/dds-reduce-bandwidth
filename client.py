import logging
import os
import shutil
import requests
import json
from dds_utils import (Results, read_results_dict, cleanup, Region,
                       compute_regions_size, extract_images_from_video,
                       merge_boxes_in_results, compute_area_of_regions)
import yaml


class Client:
    """The client of the DDS protocol
       sends images in low resolution and waits for
       further instructions from the server. And finally receives results
       Note: All frame ranges are half open ranges"""

    def __init__(self, hname, config, server_handle=None):
        if hname:
            self.hname = hname
            self.session = requests.Session()
        else:
            self.server = server_handle
        self.config = config

        self.logger = logging.getLogger("client")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.logger.info(f"Client initialized")

    def analyze_video_mpeg(self, video_name, raw_images_path, enforce_iframes):
        number_of_frames = len(
            [f for f in os.listdir(raw_images_path) if ".png" in f])

        final_results = Results()
        final_rpn_results = Results()
        total_size = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_frame = i
            end_frame = min(number_of_frames, i + self.config.batch_size)

            batch_fnames = sorted([f"{str(idx).zfill(10)}.png"
                                   for idx in range(start_frame, end_frame)])

            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{video_name}-base-phase", raw_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"{batch_video_size / 1024}KB sent "
                             f"in base phase using {self.config.low_qp}QP")
            extract_images_from_video(f"{video_name}-base-phase-cropped",
                                      req_regions)
            results, rpn_results = (
                self.server.perform_detection(
                    f"{video_name}-base-phase-cropped",
                    self.config.low_resolution, batch_fnames))

            self.logger.info(f"Detection {len(results)} regions for "
                             f"batch {start_frame} to {end_frame} with a "
                             f"total size of {batch_video_size / 1024}KB")
            final_results.combine_results(
                results, self.config.intersection_threshold)
            final_rpn_results.combine_results(
                rpn_results, self.config.intersection_threshold)

            # Remove encoded video manually
            shutil.rmtree(f"{video_name}-base-phase-cropped")
            total_size += batch_video_size

        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        final_results.fill_gaps(number_of_frames)

        # Add RPN regions
        final_results.combine_results(
            final_rpn_results, self.config.intersection_threshold)

        final_results.write(video_name)

        return final_results, [total_size, 0]

    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            # Low resolution phase
            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size


        return final_results, [total_size, 0]

    def filter_low_phase_results(self, start_fid, end_fid, low_res, fid_in_high):
        ret = Results()
        for fid in range(start_fid, end_fid):
            if fid in fid_in_high: continue
            if fid not in low_res.regions_dict: continue
            ret.regions_dict[fid] = []
            for r in low_res.regions_dict[fid]:
                ret.regions.append(r)
                ret.regions_dict[fid].append(r)
        return ret

    def analyze_video_emulate_iterative(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None,
                              debug_mode=False):
        self.server.start_server_iter()

        file1 = open(f"out/{self.config.raw_video_name}-{self.config.low_qp}-plot.txt",'w')
        file2 = open(f"out/{self.config.raw_video_name}-{self.config.low_qp}-ratio.txt",'w')
        file3 = open(f"out/{self.config.raw_video_name}-{self.config.low_qp}-ratio-shrink.txt",'w')

        self.server.init_iter_info()

        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()

        number_of_frames = len(
            [x for x in os.listdir(high_images_path) if "png" in x])

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0
        pre_size = 0

        shrink_max = 25
        enlarge_max = 40

        for i in range(0, number_of_frames, self.config.batch_size):
        # for i in range(0, 30, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            # Encode frames in batch and get size
            # Make temporary frames to downsize complete frames

            # Low resolution phase
            MAX_RATIO = shrink_max + enlarge_max
            min_fn = 1000000000

            for iter_ratio_percent in range(0, MAX_RATIO, 1):
                base_req_regions = Results()
                for fid in range(start_fid, end_fid):
                    base_req_regions.append(
                        Region(fid, 0, 0, 1, 1, 1.0, 2,
                               self.config.high_resolution))
                encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                    base_req_regions, f"{video_name}-base-phase", high_images_path,
                    self.config.low_resolution, self.config.low_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                                 f"in base phase")

                iter_ratio = 0.005 * iter_ratio_percent
                # self.logger.info(f"Testing ratio {iter_ratio}")
                low_images_path = f"{video_name}-base-phase-cropped"#-{iter_ratio_percent}"
                # self.logger.info(f"low_image_path: {low_images_path}")

                r1, req_regions = self.server.simulate_low_query(
                    start_fid, end_fid, low_images_path, low_results_dict, False,
                    self.config.rpn_enlarge_ratio, True)

                pixel_percentage = compute_area_of_regions(req_regions)

                # self.logger.info(f"{pixel_percentage}")

                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True, iter_ratio_percent)

                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")

                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions, iter_ratio_percent)
                
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                r3 = Results()
                r3.combine_results(r2, self.config.intersection_threshold)

                fids_in_r2 = [e.fid for e in r2.regions]
                r1_add = self.filter_low_phase_results(start_fid, end_fid, r1, fids_in_r2)
                r3.combine_results(r1_add, self.config.intersection_threshold)

                r3.write(f"{video_name}_{i}_{iter_ratio_percent}")
                r3 = read_results_dict(f"{video_name}_{i}_{iter_ratio_percent}")
                r3 = merge_boxes_in_results(r3, 0.3, 0.3)

                (tp, fp, fn, f1) = self.server.run_eval(r3, start_fid, end_fid)

                if fn < min_fn:
                    min_fn = fn
                    self.server.archive_iter_ratio()

                self.server.update_iter_ratio(r3, start_fid, end_fid)

                file1.write(f"{start_fid},{tp},{fp},{fn},{f1},{encoded_batch_video_size},{regions_size},{pixel_percentage}\n")
                self.logger.info(f"({iter_ratio_percent}),Eval,{tp},{fp},{fn},{f1},{encoded_batch_video_size},{regions_size},{pixel_percentage}")
                # Cleanup for the next batch
                cleanup(video_name, debug_mode, start_fid, end_fid, iter_ratio_percent)
                if iter_ratio_percent == shrink_max:
                    self.server.save_best_iter_ratio(file3)
                if fn == 0 or (iter_ratio_percent>40 and abs(regions_size-pre_size)<1):
                    break
                pre_size = regions_size

            self.server.load_iter_ratio()
            # _ = self.server.save_iter_ratio()
            self.logger.info(f"min fn = {min_fn}")

            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            self.logger.info(f"Sent {encoded_batch_video_size / 1024} "
                             f"in base phase")
            total_size[0] += encoded_batch_video_size

            low_images_path = f"{video_name}-base-phase-cropped"
            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(
                r1, self.config.intersection_threshold)
            # final_results.combine_results(
            #     r1, self.config.intersection_threshold)

            # High resolution phase
            if len(req_regions) > 0:
                # Crop, compress and get size
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(f"Sent {len(req_regions)} regions which have "
                                 f"{regions_size / 1024}KB in second phase "
                                 f"using {self.config.high_qp}")
                total_size[1] += regions_size

                # High resolution phase every three filter
                r2 = self.server.emulate_high_query(
                    video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase "
                                 f"of batch")

                high_phase_results.combine_results(
                    r2, self.config.intersection_threshold)
                final_results.combine_results(
                    r2, self.config.intersection_threshold)

                fids_in_r2 = [e.fid for e in r2.regions]
                r1_add = self.filter_low_phase_results(start_fid, end_fid, r1, fids_in_r2)
                # r3.combine_results(r1_add, self.config.intersection_threshold)


            # low_phase_results.combine_results(
            #     r1, self.config.intersection_threshold)
            final_results.combine_results(
                r1_add, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(video_name, debug_mode, start_fid, end_fid)

        self.server.save_iter_ratio(file2)

        self.logger.info(f"Got {len(low_phase_results)} unique results "
                         f"in base phase")
        self.logger.info(f"Got {len(high_phase_results)} positive "
                         f"identifications out of {total_regions_count} "
                         f"requests in second phase")

        # Fill gaps in results
        final_results.fill_gaps(number_of_frames)

        # Write results
        final_results.write(f"{video_name}")

        self.logger.info(f"Writing results for {video_name}")
        self.logger.info(f"{len(final_results)} objects detected "
                         f"and {total_size[1]} total size "
                         f"of regions sent in high resolution")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")
        return final_results, total_size


    def init_server(self, nframes):
        self.config['nframes'] = nframes
        response = self.session.post(
            "http://" + self.hname + "/init", data=yaml.dump(self.config))
        if response.status_code != 200:
            self.logger.fatal("Could not initialize server")
            # Need to add exception handling
            exit()

    def get_first_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(
            vid_name + "-base-phase-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/low", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))
        rpn = Results()
        for region in response_json["req_regions"]:
            rpn.append(Region.convert_from_server_response(
                region, self.config.low_resolution, "low-res"))

        return results, rpn

    def get_second_phase_results(self, vid_name):
        encoded_vid_path = os.path.join(vid_name + "-cropped", "temp.mp4")
        video_to_send = {"media": open(encoded_vid_path, "rb")}
        response = self.session.post(
            "http://" + self.hname + "/high", files=video_to_send)
        response_json = json.loads(response.text)

        results = Results()
        for region in response_json["results"]:
            results.append(Region.convert_from_server_response(
                region, self.config.high_resolution, "high-res"))

        return results

    def analyze_video(
            self, vid_name, raw_images, config, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0
        nframes = sum(map(lambda e: "png" in e, os.listdir(raw_images)))

        self.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(
                    fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ = compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            results, rpn_regions = self.get_first_phase_results(vid_name)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(rpn_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name)
                final_results.combine_results(
                    results, self.config.intersection_threshold)

            # Cleanup for the next batch
            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)
