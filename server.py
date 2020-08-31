import os
import shutil
import logging
import cv2 as cv
from dds_utils import (Results, Region, calc_iou, merge_images,
                       extract_images_from_video, merge_boxes_in_results,
                       compute_area_of_frame, calc_area, read_results_dict)
from .object_detector import Detector


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""
    
    # Add by shibiao
    def _make_dict(self, path):
        now_dict = {}
        with open(path, "r") as f:
            for line in f.readlines():
                data = line.strip().split(',')
                assert(len(data) == 8, "The regions in the file " + path + " is not correct.")

                # The order of [label] and [confidence] in the file is different that in the constructor. 
                data[5], data[6] = data[6], data[5]

                region = Region(*data)
                if not region.fid in now_dict:
                    now_dict[region.fid] = []
                if region.conf < 0.5 or region.w * region.h > 0.04 or region.label != "vehicle" or region.origin == "generic":
                    continue
                now_dict[region.fid].append(region)
        return now_dict

    def read_ratio(self, path):
        now_dict = {}
        with open(path, "r") as f:
            for line in f.readlines():
                data = line.strip().split(',')
                # assert(len(data) == 6, "The regions in the file " + path + " is not correct.")
                region = Region(*data)
                if not region.fid in now_dict:
                    now_dict[region.fid] = []
                now_dict[region.fid].append(region)
        return now_dict

    # Modify by shibiao
    def __init__(self, config, nframes=None):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)

        self.detector = Detector()

        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None

        self.logger.info("Server started")

    def start_server_iter(self):
        self.load_mpeg_result()
        self.ratio_results = self._make_dict(f"results/{self.config.raw_video_name}_mpeg_1.0_26")

        self.iter_1 = {}
        self.iter_2 = {}
        for fid in self.ratio_results:
            self.iter_1[fid] = {}
            self.iter_2[fid] = {}
            for region in self.ratio_results[fid]:
                # self.logger.info(f"{fid}")
                self.iter_1[fid][region] = 0
                self.iter_2[fid][region] = False
        self.logger.info("Server started (iterative)")

    def load_mpeg_result(self):
        self.high_configuration_results = self._make_dict(f"results/{self.config.raw_video_name}_mpeg_1.0_26")
        self.low_configuration_results = self._make_dict(f"results/{self.config.raw_video_name}_mpeg_1.0_{self.config.low_qp}")

    def run_eval(self, results, start_fid, end_fid, record=False):
        self.logger.info(f"Count detect fail")
        detect_fail = 0
        tp = 0
        fp = 0
        fn = 0
        for fid in range(start_fid, end_fid):
            for r1 in self.high_configuration_results[fid]:
                # if self.iter_2[fid][r1] == True:
                #     continue
                detected = False
                if fid in results.regions_dict:
                    for r2 in results.regions_dict[fid]:
                        if r2.conf < 0.5 or r2.w * r2.h > 0.04 or r2.label != "vehicle" or r2.origin == "generic":
                            continue
                        if calc_iou(r1, r2) > 0.3:#self.config.objfilter_iou:
                            detected = True
                            break
                if detected == True:
                    continue
                fn += 1

            if fid in results.regions_dict:
                for r2 in results.regions_dict[fid]:
                    if r2.conf < 0.5 or r2.w * r2.h > 0.04 or r2.label != "vehicle" or r2.origin == "generic":
                        continue
                    detected = False
                    for r1 in self.high_configuration_results[fid]:
                        if calc_iou(r1, r2) > 0.3:
                            detected = True
                            break
                    if detected == True:
                        tp += 1
                    else:
                        fp += 1

        return (tp, fp, fn, round((2.0*tp)/(2.0*tp+fp+fn),3))


    def update_iter_ratio(self, results, start_fid, end_fid, record=False):
        self.logger.info(f"Update iter ratio")
        for fid in range(start_fid, end_fid):
            for r1 in self.ratio_results[fid]:
                detected = False
                if fid in results.regions_dict:
                    for r2 in results.regions_dict[fid]:
                        if r2.conf < 0.5 or r2.w * r2.h > 0.04 or r2.label != "vehicle" or r2.origin == "generic":
                            continue
                        if calc_iou(r1, r2) > 0.3:
                            detected = True
                            self.iter_2[fid][r1] = True
                            break
                if detected == True:
                    continue
                # self.logger.info(f"Increase ratio {fid},{r1.x},{r1.y},{r1.w},{r1.h}")
                self.iter_1[fid][r1] += 1
                self.iter_2[fid][r1] = False

    def save_best_iter_ratio(self, file):
        for fid in self.ratio_results:
            for region in self.ratio_results[fid]:
                t = 0.0
                if self.best_iter_2[fid][region] == True:
                    t = 1.0
                file.write(f"{fid},{region.x},{region.y},{region.w},{region.h},{self.best_iter_1[fid][region]},{region.label},{t}\n")

    def save_iter_ratio(self, file):
        for fid in self.ratio_results:
            for region in self.ratio_results[fid]:
                t = 0.0
                if self.iter_2[fid][region] == True:
                    t = 1.0
                file.write(f"{fid},{region.x},{region.y},{region.w},{region.h},{self.iter_1[fid][region]},{region.label},{t}\n")

    def archive_iter_ratio(self):
        self.logger.info("Archive iter ratio")
        self.best_iter_1 = {}
        self.best_iter_2 = {}
        for fid in self.iter_1:
            self.best_iter_1[fid] = {}
            self.best_iter_2[fid] = {}
            for r1 in self.iter_1[fid]:
                self.best_iter_1[fid][r1] = self.iter_1[fid][r1]
                self.best_iter_2[fid][r1] = self.iter_2[fid][r1]
                # self.logger.info(f"{fid},{r1.x},{r1.y},{r1.w},{r1.h},{self.best_iter_1[fid][r1]},{self.best_iter_2[fid][r1]}")


    def load_iter_ratio(self):
        self.logger.info("Load iter ratio")
        self.iter_1 = {}
        self.iter_2 = {}
        for fid in self.best_iter_1:
            self.iter_1[fid] = {}
            self.iter_2[fid] = {}
            for r1 in self.best_iter_1[fid]:
                self.iter_1[fid][r1] = self.best_iter_1[fid][r1]
                self.iter_2[fid][r1] = self.best_iter_2[fid][r1]
                # self.logger.info(f"{fid},{r1.x},{r1.y},{r1.w},{r1.h},{self.iter_1[fid][r1]},{self.iter_2[fid][r1]}")

    def init_iter_info(self):
        self.logger.info(f"Iterative ratio initialized")
        self.iter_1 = {}
        self.iter_2 = {}
        for fid in self.ratio_results:
            self.iter_1[fid] = {}
            self.iter_2[fid] = {}
            for region in self.ratio_results[fid]:
                # if fid<5:
                #     self.logger.info(f"{fid}, {region.x}, {region.y}, {region.w}, {region.h}")
                self.iter_1[fid][region] = 0
                self.iter_2[fid][region] = False

    def reset_state(self, nframes):
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def perform_server_cleanup(self):
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def perform_detection(self, images_direc, resolution, fnames=None,
                          images=None):
        final_results = Results()
        rpn_regions = Results()

        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            detection_results, rpn_results = (
                self.detector.infer(image))
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                if (self.config.min_object_size and
                        w * h < self.config.min_object_size) or w * h == 0.0:
                    continue
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                final_results.append(r)
                frame_with_no_results = False
            for label, conf, (x, y, w, h) in rpn_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="generic")
                rpn_regions.append(r)
                frame_with_no_results = False
            self.logger.debug(
                f"Got {len(final_results)} results "
                f"and {len(rpn_regions)} for {fname}")

            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        return final_results, rpn_regions

    def get_regions_to_query(self, rpn_regions, detections):
        req_regions = Results()
        for region in rpn_regions.regions:
            # Continue if the size of region is too large
            if region.w * region.h > self.config.size_obj:
                continue

            # If there are positive detections and they match a region
            # skip that region
            if len(detections) > 0:
                matches = 0
                for detection in detections.regions:
                    if (calc_iou(detection, region) >
                            self.config.objfilter_iou and
                            detection.fid == region.fid and
                            region.label == 'object'):
                        matches += 1
                if matches > 0:
                    continue

            # Enlarge and add to regions to be queried
            region.enlarge(self.config.rpn_enlarge_ratio)
            req_regions.add_single_result(
                region, self.config.intersection_threshold)
        return req_regions
    
    def query_min_enlarge_ratio(self, fid, tx, ty):
        for region in self.ratio_results[fid]:
            # self.logger.info(f"{region.x} {region.y} {region.w} {region.h} {region.conf}")
            if abs(tx-region.x)<1e-6 and abs(ty-region.y)<1e-6:
                return region.conf
        return -1

    def query_iter_enlarge_ratio(self, fid, tx, ty, region_q=None):
        for region in self.ratio_results[fid]:
            # self.logger.info(f"{region.x} {region.y} {region.w} {region.h} {region.conf}")
            if abs(tx-region.x)<1e-6 and abs(ty-region.y)<1e-6:
                return self.iter_1[fid][region]
        self.logger.info(f"FAIL {fid}, {tx}, {ty}")
        self.ratio_results[fid].append(region_q)
        self.iter_1[fid][region_q] = 0
        self.iter_2[fid][region_q] = False

        # assert(False)
        return 0

    # Add by shibiao
    def get_regions_to_query_new(self, start_fid, end_fid, low_configuration_results, high_configuration_results, shrink_max=25):
        '''
        Args: 
            all_regions [list] a list of regions.
            low_resolution_results [dict] a dict of regions using 'fid' as key.
            high_resolution_results [dict] a dict of regions using 'fid' as key.
        Returns:
            req_regions [list] a list of regions detected in low configuration but are not detected in high configuration.
        '''

        # self.logger.info(f"Running get_regions_to_query_new")
        # for fid in range(start_fid, end_fid):
        #     for high_detection in high_configuration_results[fid]:
        #         self.logger.info(f"{fid} {high_detection.x} {high_detection.y} {high_detection.w} {high_detection.h} {high_detection.label} {high_detection.origin}")
        self.load_mpeg_result()
        req_regions = Results()

        shrink_r = [(1.0/(shrink_max**0.5))*i**0.5 for i in range(1, shrink_max+1)]

        for fid in range(start_fid, end_fid):
            # single_regions = Results()
            for high_detection in high_configuration_results[fid]:
                # self.logger.info(f"Checking {fid} {high_detection.x} {high_detection.y} {high_detection.w} {high_detection.h}")
                has_overlap = False
                for low_detection in low_configuration_results[fid]:
                    if calc_iou(high_detection, low_detection) > self.config.objfilter_iou:
                        # self.logger.info(f"Overlap with {low_detection.x} {low_detection.y} {low_detection.w} {low_detection.h}")
                        has_overlap = True
                        break
                if has_overlap == False:
                    # occur only in high quality reference, need to return
                    # self.logger.info(f"{fid},{high_detection.x},{high_detection.y},{high_detection.w},{high_detection.h},{high_detection.conf},{high_detection.label},{high_detection.resolution}")

                    ratio = self.query_iter_enlarge_ratio(fid, high_detection.x, high_detection.y, high_detection)
                    if ratio<0:
                        # self.logger.info(f"No overlap, min enlarge ratio failed")
                        continue
                    # self.logger.info(f"Enlarge ratio {ratio}")
                    if ratio >= shrink_max:
                        ratio -= (shrink_max - 1)
                        ratio *= 0.005
                    else:
                        ratio = - shrink_r[ratio]

                    # ratio *= 0.01
                    # ratio = 1.0

                    self.logger.info(f"{fid} {high_detection.x} {high_detection.y} {high_detection.w} {high_detection.h} {ratio}")

                    # xx, yy, ww, hh = high_detection.x, high_detection.y, high_detection.w, high_detection.h
                    high_detection.enlarge_absolute(ratio)
                    # high_detection.enlarge_absolute(self.config.rpn_enlarge_ratio)

                    # self.logger.info(f"{fid} {high_detection.x} {high_detection.y} {high_detection.w} {high_detection.h} {ratio}")

                    req_regions.add_single_result(
                        high_detection, self.config.intersection_threshold)
                    
                    # single_regions.add_single_result(high_detection, self.config.intersection_threshold)
                    # high_detection.x, high_detection.y, high_detection.w, high_detection.h = xx, yy, ww, hh



        return req_regions

    # Modify by shibiao
    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        if extract_regions:
            # If called from actual implementation
            # This will not run
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            extract_images_from_video(images_direc, base_req_regions)

        batch_results = Results()

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")
        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.config.intersection_threshold)

        detections = Results()
        # rpn_regions = Results()
        # Divide RPN results into detections and RPN regions
        for single_result in batch_results.regions:
            if (single_result.conf > self.config.prune_score and
                    single_result.label == "vehicle"):
                detections.add_single_result(
                    single_result, self.config.intersection_threshold)
            # else:
                # rpn_regions.add_single_result(
                    # single_result, self.config.intersection_threshold)

        #regions_to_query = self.get_regions_to_query(rpn_regions, detections)
        regions_to_query = self.get_regions_to_query_new(start_fid, end_fid, self.low_configuration_results, self.high_configuration_results)

        return detections, regions_to_query

    def emulate_high_query(self, vid_name, low_images_direc, req_regions, iter=None):
        images_direc = f"{vid_name}-cropped"#-{iter}"
        # self.logger.info(f"images_direc: {images_direc}")
        # Extract images from encoded video
        extract_images_from_video(images_direc, req_regions)

        if not os.path.isdir(images_direc):
            self.logger.error("Images directory was not found but the "
                              "second iteration was called anyway")
            return Results()

        fnames = sorted([f for f in os.listdir(images_direc) if "png" in f])

        # Make seperate directory and copy all images to that directory
        merged_images_direc = os.path.join(images_direc, "merged")
        os.makedirs(merged_images_direc, exist_ok=True)
        for img in fnames:
            shutil.copy(os.path.join(images_direc, img), merged_images_direc)

        # self.logger.info(f"{merged_images_direc}, {low_images_direc}")

        merged_images = merge_images(
            merged_images_direc, low_images_direc, req_regions)
        results, _ = self.perform_detection(
            merged_images_direc, self.config.high_resolution, fnames,
            merged_images)

        results_with_detections_only = Results()
        for r in results.regions:
            if r.label == "no obj":
                continue
            results_with_detections_only.add_single_result(
                r, self.config.intersection_threshold)

        high_only_results = Results()
        area_dict = {}
        for r in results_with_detections_only.regions:
            frame_regions = req_regions.regions_dict[r.fid]
            regions_area = 0
            if r.fid in area_dict:
                regions_area = area_dict[r.fid]
            else:
                regions_area = compute_area_of_frame(frame_regions)
                area_dict[r.fid] = regions_area
            regions_with_result = frame_regions + [r]
            total_area = compute_area_of_frame(regions_with_result)
            extra_area = total_area - regions_area
            if extra_area < 0.05 * calc_area(r):
                r.origin = "high-res"
                high_only_results.append(r)

        # shutil.rmtree(merged_images_direc)

        return results_with_detections_only

    def perform_low_query(self, vid_data):
        # Write video to file
        with open(os.path.join("server_temp", "temp.mp4"), "wb") as f:
            f.write(vid_data.read())

        # Extract images
        # Make req regions for extraction
        start_fid = self.curr_fid
        end_fid = min(self.curr_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"Processing frames from {start_fid} to {end_fid}")
        req_regions = Results()
        for fid in range(start_fid, end_fid):
            req_regions.append(
                Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        extract_images_from_video("server_temp", req_regions)
        fnames = [f for f in os.listdir("server_temp") if "png" in f]

        results, rpn = self.perform_detection(
            "server_temp", self.config.low_resolution, fnames)

        batch_results = Results()
        batch_results.combine_results(
            results, self.config.intersection_threshold)

        # need to merge this because all previous experiments assumed
        # that low (mpeg) results are already merged
        batch_results = merge_boxes_in_results(
            batch_results.regions_dict, 0.3, 0.3)

        batch_results.combine_results(
            rpn, self.config.intersection_threshold)

        detections, regions_to_query = self.simulate_low_query(
            start_fid, end_fid, "server_temp", batch_results.regions_dict,
            False, self.config.rpn_enlarge_ratio, False)

        self.last_requested_regions = regions_to_query
        self.curr_fid = end_fid

        # Make dictionary to be sent back
        detections_list = []
        for r in detections.regions:
            detections_list.append(
                [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
        req_regions_list = []
        for r in regions_to_query.regions:
            req_regions_list.append(
                [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        return {
            "results": detections_list,
            "req_regions": req_regions_list
        }

    def perform_high_query(self, file_data):
        low_images_direc = "server_temp"
        cropped_images_direc = "server_temp-cropped"

        with open(os.path.join(cropped_images_direc, "temp.mp4"), "wb") as f:
            f.write(file_data.read())

        results = self.emulate_high_query(
            low_images_direc, low_images_direc, self.last_requested_regions)

        results_list = []
        for r in results.regions:
            results_list.append([r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])

        # Perform server side cleanup for the next batch
        self.perform_server_cleanup()

        return {
            "results": results_list,
            "req_region": []
        }