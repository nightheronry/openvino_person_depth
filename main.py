#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IECore


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=False, type=str, default="models\person-detection-retail-0013\FP32\person-detection-retail-0013.xml")
    parser.add_argument(
        "-dm", "--depth_model", help="Required. Path to an .xml file with a trained model", required=False, type=str, default="models\midasnet/FP32/midasnet-v2.xml")
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or image. 'cam' for capturing video stream from camera",
                      required=False, type=str, default="cam")
    args.add_argument("-o", "--output",
                      help="",
                      required=False, type=str, default="")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.ERROR, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Creating Inference Engine...")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    ie_d = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie_d.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network")
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")
    ###depth
    depth_net = ie_d.read_network(args.depth_model, os.path.splitext(args.depth_model)[0] + ".bin")
    if "CPU" in args.device:
        supported_layers = ie_d.query_network(depth_net, "CPU")
        not_supported_layers = [l for l in depth_net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    ###
    if "CPU" in args.device:
        supported_layers = ie_d.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))

    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    out_blob = next(iter(net.outputs))
    #####  depth####

    depth_input_blob = next(iter(depth_net.inputs))
    depth_out_blob = next(iter(depth_net.outputs))
    depth_net.batch_size = 1

    ###############

    log.info("Loading IR to the plugin...")
    exec_net = ie.load_network(network=net, num_requests=2, device_name=args.device)

    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    ### depth
    exec_d_net = ie_d.load_network(network=depth_net, device_name=args.device)
    _, _, d_height, d_width = depth_net.inputs[depth_input_blob].shape
    ###

    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]

    if args.input == 'cam':
        input_stream = 0

        cap = cv2.VideoCapture(input_stream)
        assert cap.isOpened(), "Can't open " + input_stream

        cur_request_id = 0
        d_cur_request_id = 0


        log.info("Starting inference in anasync mode...")

        render_time = 0

        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        print("To switch between rgb/depth modes, press TAB key in the output window")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_h, frame_w = frame.shape[:-1]
            if not ret:
                break  # abandons the last frame in case of async_mode
            # Main sync point:
            # in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            # in the regular mode we start the CURRENT request and immediately wait for it's completion
            inf_start = time.time()
            in_frame = cv2.resize(frame, (w, h))
            d_in_frame = in_frame.copy()
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
            #depth
            (d_input_height, d_input_width) = d_in_frame.shape[:-1]
            # resize
            if (d_input_height, d_input_width) != (d_height, d_width):
                log.info("Image is resized from {} to {}".format(
                    d_in_frame.shape[:-1], (d_height, d_width)))
                d_in_frame = cv2.resize(d_in_frame, (d_width, d_height), cv2.INTER_CUBIC)

            ###
            #depth
            d_in_frame = d_in_frame.transpose((2, 0, 1))
            image_input = np.expand_dims(d_in_frame, 0)
            exec_d_net.start_async(request_id=d_cur_request_id, inputs={depth_input_blob: image_input})
            d_signal = exec_d_net.requests[d_cur_request_id].wait(-1) == 0
            signal = exec_net.requests[cur_request_id].wait(-1) == 0
            if d_signal and signal:
                inf_end = time.time()
                det_time = inf_end - inf_start

                # Parse detection results of the current request
                res = exec_net.requests[cur_request_id].outputs[out_blob]

                ###depth
                disp = exec_d_net.requests[d_cur_request_id].outputs[depth_out_blob][0]
                disp = cv2.resize(disp, (frame_w, frame_h), cv2.INTER_CUBIC)

                disp_min = disp.min()
                disp_max = disp.max()
                depth_ref = (disp[int(frame_w/2)][int(frame_h/2)])
                ###
                if args.output == 'depth':
                    if disp_max - disp_min > 1e-6:
                        disp_output = (disp - disp_min) / (disp_max - disp_min)
                    else:
                        disp_output.fill(0.5)
                    frame = cv2.cvtColor(disp_output, cv2.COLOR_GRAY2BGR)
                for obj in res[0][0]:
                    # Draw only objects when probability more than specified threshold
                    if obj[2] > args.prob_threshold:
                        xmin = int(obj[3] * frame_w)
                        ymin = int(obj[4] * frame_h)
                        xmax = int(obj[5] * frame_w)
                        ymax = int(obj[6] * frame_h)

                        # depth
                        depth = float(15000.0-disp[int((ymin+((ymax+ymin)/2))/2), int((xmax+xmin)/2)])/10000.0
                        ###

                        color = (0, 0, 255)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, f'{round(round(depth, 4)*10.0-10.0, 2)}' + 'm ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            # Draw performance stats
            inf_time_message = "Inference time: {:.3f} ms".format(det_time * 1000)
            render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
            async_mode_message = "Async mode is off. Processing request {}".format(cur_request_id)
            depth_message = "depth of reference point: {}".format(depth_ref)

            cv2.rectangle(frame, (int(frame_w/2), int(frame_h/2)), (int(frame_w/2), int(frame_h/2)), (0, 255, 255), 20)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
            cv2.putText(frame, async_mode_message, (10, int(frame_h - 20)), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (10, 10, 200), 1)
            cv2.putText(frame, depth_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            render_start = time.time()
            if not args.no_show:
                cv2.imshow("Detection Results", frame)
            render_end = time.time()
            render_time = render_end - render_start

            if not args.no_show:
                key = cv2.waitKey(1)
                if key == 27:
                    break
                if 9 == key:
                    args.output = "rgb" if args.output is "depth" else "depth"
                    log.debug("Switched to {} mode".format("rgb" if args.output else "depth"))

        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
