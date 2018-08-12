#!/usr/bin/env python

"""Example of how to use ipm"""

import ipm


def test(matcher, image_path):
    print("Testing: " + image_path)

    patterns = matcher.match(path=image_path)
    print("Matched: " + patterns[0].name)


def main():
    m = ipm.Matcher()
    m.add_pattern(path='res/logo_drpepper.jpg', name='Dr Pepper')
    m.add_pattern(path='res/logo_pepsi.jpg', name='Pepsi')

    test(matcher=m, image_path='res/can_pepsi.jpg')
    test(matcher=m, image_path='res/can_drpepper.jpg')
    test(matcher=m, image_path='res/can_drpepper_cherry.jpg')


if __name__ == '__main__':
    main()
