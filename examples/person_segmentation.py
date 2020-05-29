import argparse
import sys

import cv2
from albumentations import Compose, SmallestMaxSize, CenterCrop
from pietoolbelt.viz import ColormapVisualizer

from segmentation import Segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation example')
    parser.add_argument('-i', '--image', type=str, help='Path to image to predict', required=False)
    parser.add_argument('-w', '--web_cam', help='Use web camera id to predict', action='store_true')
    parser.add_argument('-d', '--device', type=str, help='Device', required=False, default='cuda')

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    if (args.image is None and args.web_cam is None) or (args.image is not None and args.web_cam is not None):
        print("Please define one of option: -i or -w")
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    vis = ColormapVisualizer([0.5, 0.5])

    seg = Segmentation(accuracy_lvl=Segmentation.Level.LEVEL_2)
    seg.set_device(args.device)

    data_transform = Compose([SmallestMaxSize(max_size=512, always_apply=True),
                              CenterCrop(height=512, width=512, always_apply=True)], p=1)

    if args.image is not None:
        image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        image = data_transform(image)
        cv2.imwrite('result.jpg', seg.process(image)[0])

    elif args.web_cam is not None:
        title = "Person segmentation example"

        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cap = cv2.VideoCapture(0)

        while cv2.waitKey(1) & 0xFF != ord('q'):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = data_transform(image=frame)['image']

            img, mask = seg.process(frame)
            image = vis.process_img(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask)
            cv2.imshow(title, image)

        cap.release()
        cv2.destroyAllWindows()
