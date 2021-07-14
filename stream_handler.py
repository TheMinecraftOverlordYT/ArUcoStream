import streamlink
import cv2

url = 'twitch.tv/qtcinderella'


def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)

    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError('No streams were available.')


def get_stream():
    cap = cv2.VideoCapture(stream_to_url(url, '480p'))
    return cap
