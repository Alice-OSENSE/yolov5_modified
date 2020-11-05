import cv2


def stream_result(img, scale, mode='video', window_name='result'):
    """
    img (numpy.ndarray)
    scale (float)
    mode (str)
    window_name (str)
    """
    im0_resized = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow(window_name, im0_resized)
    if mode == 'video':
        cv2.waitKey(delay=1)
    elif cv2.waitKey(1) == ord('q'):  # q to quit
        raise StopIteration