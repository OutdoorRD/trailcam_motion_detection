"""Trailcam motion detection example."""
import os
import time

import cv2

MIN_CONTOUR_AREA = 4500
VIDEO_PATH = r"C:\Users\Daddy\Downloads\160828AB.avi"
MASK_PATH = r"E:\repositories\trailcam_motion\160828AB.avi_mask.png"
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
    filter_frame = None
    image_index = -1
    motion_in_last_frame = False
    mask = cv2.imread(MASK_PATH)
    consecutive_start = -1
    stats = []
    n_detected = 0
    while True:
        image_index += 1
        valid, frame = video.read()
        if not valid:
            break

        masked_frame = cv2.bitwise_and(frame, mask)
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
        detected_hiker = False
        x, y, w, h = None, None, None, None
        for contour in contour_list:
            # if the contour is too small, ignore it
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            if x is None:
                (x, y, w, h) = cv2.boundingRect(contour)
                xp = x + w
                yp = y + h
            else:
                (local_x, local_y, local_w, local_h) = cv2.boundingRect(
                    contour)
                local_xp = local_x + local_w
                local_yp = local_y + local_h

                x = min(x, local_x)
                xp = max(xp, local_xp)
                y = min(y, local_y)
                yp = max(yp, local_yp)
            detected_hiker = True

        if detected_hiker:
            w = xp - x
            h = yp - y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            image_path = os.path.join(
                output_dir, basename + '_%d.png' % image_index)
            cv2.imwrite(image_path, frame)
            if consecutive_start == -1:
                consecutive_start = image_index
            n_detected += 1
        else:
            if consecutive_start != -1:
                stats.append((consecutive_start, image_index-1))
                print 'consecutive series: %d - %d, %d' % (
                    consecutive_start, image_index - 1,
                    image_index - consecutive_start)
                consecutive_start = -1
        last_filter_frame = filter_frame

    video.release()
    cv2.destroyAllWindows()

    stats_path = os.path.join(OUTPUT_BASE_DIR, basename + '_stats.txt')
    with open(stats_path, 'w') as statsfile:
        statsfile.write("%d frames detected\n" % n_detected)
        statsfile.write("%d series detected\n" % len(stats))
        statsfile.write("Consecutive Frame series:\n")
        for start_index, end_index in stats:
            statsfile.write(
                "%d - %d, %d\n" % (
                    start_index, end_index, end_index - start_index + 1))
    print open(stats_path, 'r').read()


if __name__ == '__main__':
    main()

