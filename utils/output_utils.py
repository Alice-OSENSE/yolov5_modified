import cv2

# called if save_path is not vid_path
def get_new_video_writer(save_path, vid_writer=None, vid_cap=None, rotate=False):
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release previous video writer

    fourcc = 'mp4v'  # output video codec
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotate:
        w, h = h, w
    return cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))