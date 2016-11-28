"""Trailcam motion detection example with OpenCV and mask."""
import os
import time

import cv2

MIN_CONTOUR_AREA = 4000
VIDEO_PATH = r"./160913AA.TLV"
AOI_MASK_PATH = r"./160913AA.TLV_mask.png"
OUTPUT_BASE_DIR = 'trailcam_detection_results'


def main():
    """Entry point."""
    # make an output folder
    basename = os.path.basename(VIDEO_PATH)
    output_dir = os.path.join(OUTPUT_BASE_DIR, os.path.splitext(basename)[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load video object
    video = cv2.VideoCapture(VIDEO_PATH)
    last_filter_frame = None
    image_index = -1
    aoi_mask = cv2.imread(AOI_MASK_PATH)
    consecutive_start = -1
    series_list = []
    n_detected = 0
    while True:
        image_index += 1
        valid, frame = video.read()
        if not valid:
            break

        masked_frame = cv2.bitwise_and(frame, aoi_mask)
        # resize the frame, convert it to grayscale, and blur it
        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        filter_frame = cv2.GaussianBlur(gray_frame, (31, 31), 0)

        if last_filter_frame is None:
            last_filter_frame = filter_frame

        delta = cv2.absdiff(last_filter_frame, filter_frame)
        thresholded_frame = cv2.threshold(
            delta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        dilated_frame = cv2.dilate(thresholded_frame, None, iterations=3)
        (contour_list, _) = cv2.findContours(
            dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours and construct the largest bounding box
        detected_motion = False
        x, y, w, h = None, None, None, None
        for contour in contour_list:
            # if the contour is too small, ignore it
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            if x is None:
                # base bounding box
                (x, y, w, h) = cv2.boundingRect(contour)
                xp = x + w
                yp = y + h
            else:
                # calculate how this bounding box fits in with the global one
                (local_x, local_y, local_w, local_h) = cv2.boundingRect(
                    contour)
                local_xp = local_x + local_w
                local_yp = local_y + local_h
                x = min(x, local_x)
                xp = max(xp, local_xp)
                y = min(y, local_y)
                yp = max(yp, local_yp)
            detected_motion = True

        if detected_motion:
            # recalculate width and height based on BB cooridnates
            w = xp - x
            h = yp - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, str(w*h), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 0, 255), 2)
            cv2.putText(frame, str(w*h), (1, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 255), 2)
            image_path = os.path.join(
                output_dir, basename + '_%d.png' % image_index)
            # save that frame to disk
            cv2.imwrite(image_path, frame)
            # start tracking a new consecutive sequence
            if consecutive_start == -1:
                consecutive_start = image_index
            n_detected += 1
        else:
            if consecutive_start != -1:
                # if end of consecutive sequence, calculate the results
                series_list.append((consecutive_start, image_index-1))
                print 'consecutive motion series: %d - %d, %d' % (
                    consecutive_start, image_index - 1,
                    image_index - consecutive_start)
                consecutive_start = -1
        last_filter_frame = filter_frame

    video.release()
    cv2.destroyAllWindows()

    # save a nice stats file and print result
    stats_path = os.path.join(OUTPUT_BASE_DIR, basename + '_stats.txt')
    with open(stats_path, 'w') as statsfile:
        statsfile.write("%d motion frames detected\n" % n_detected)
        statsfile.write("%d series detected\n" % len(series_list))
        statsfile.write("consecutive motion series:\n")
        for start_index, end_index in series_list:
            statsfile.write(
                "%d - %d, %d\n" % (
                    start_index, end_index, end_index - start_index + 1))
    print open(stats_path, 'r').read()


if __name__ == '__main__':
    main()
