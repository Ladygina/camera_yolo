import threading
import cv2
from ultralytics import YOLO

# for video .mp4
def run_tracker_in_thread(filename, model):
    """Starts multi-thread tracking on video from `filename` using `model` and displays results frame by frame."""
    video = cv2.VideoCapture(filename)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frames):
        ret, frame = video.read()
        if ret:
            results = model.track(source=frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)
            if cv2.waitKey(1) == ord("q"):
                break
def run_tracker_camera(model):
    """Starts multi-thread tracking on video from `filename` using `model` and displays results frame by frame."""
    vid = cv2.VideoCapture(0)
    img_name = 0

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if ret:
            results = model.track(source=frame, persist=True)
            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)
            cv2.imwrite('D:\pycharm\camera\pictures\picture'+str(img_name)+'.jpg',frame)
            cv2.imwrite('D:\pycharm\camera\detect_pictures\detect'+str(img_name)+'.jpg', res_plotted)
        img_name += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

model1 = YOLO("best1.pt")
#model2 = YOLO("yolov8n-seg.pt")

# Define the video files for the trackers
video_file1 = "дрон_видео.mp4"

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1), daemon=True)
#tracker_thread1 = threading.Thread(target=run_tracker_camera(model1), daemon=True)

# Start the tracker threads
tracker_thread1.start()

# Wait for the tracker threads to finish
tracker_thread1.join()

# Clean up and close windows
cv2.destroyAllWindows()