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
            continue

        masked_frame = cv2.bitwise_and(frame, mask)
        # resize the frame, convert it to grayscale, and blur it
        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        filter_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if last_filter_frame is None:
            last_filter_frame = filter_frame

        delta = cv2.absdiff(last_filter_frame, filter_frame)
        thresholded_frame = cv2.threshold(
            delta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        dilated_frame = cv2.dilate(thresholded_frame, None, iterations=2)
        (contour_list, _) = cv2.findContours(
            dilated_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        detected_hiker = False
        for contour in contour_list:
            # if the contour is too small, ignore it
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(delta, (x, y), (x + w, y + h), (255, 255, 255), 2)
            detected_hiker = True

        if detected_hiker:
            image_path = os.path.join(
                output_dir, basename + '_%d.png' % image_index)
            print "%d,%s" % (image_index, image_path)
            cv2.imwrite(image_path, frame)
            if consecutive_start == -1:
                consecutive_start = image_index
            n_detected += 1
        else:
            if consecutive_start != -1:
                stats.append((consecutive_start, image_index))
            last_filter_frame = filter_frame

        #cv2.imshow("Raw Image", frame)
        #cv2.imshow("Threshholded Delta", thresholded_frame)
        #cv2.imshow("Frame Delta", delta)

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

        if n_detected > 20:
            break

    video.release()
    cv2.destroyAllWindows()

    with open(basename + '_stats.txt', 'w') as statsfile:
        statsfile.write("%d frames detected\n" % n_detected)
        statsfile.write("%d series detected\n" % len(stats))
        statsfile.write("Consecutive Frame series:\n")
        for start_index, end_index in stats:
            stats.write(
                "%d - %d, %d" % (
                    start_index, end_index, end_index - start_index))



if __name__ == '__main__':
    main()
