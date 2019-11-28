#!/usr/bin/env python3

from glob import glob
from time import sleep

from config import ROOT, LOG
import count


if __name__ == "__main__":
    try:
        while True:
            known_imgs = {LOG}
            imgs_in_dir = sorted(glob(ROOT + "*"))
            with open(LOG, "r") as f:
                known = f.readlines()
            for k in known:
                img, _ = k.split(":")
                img, _ = img.split("[")
                del _
                known_imgs.add(ROOT + img)
            for img_in_dir in imgs_in_dir:
                if img_in_dir not in known_imgs:
                    print("\tfound new img!        ")
                    print(f"\t{img_in_dir.split('/')[-1]}")
                    res = 2000
                    while True:
                        try:
                            print(f"\trasterize = {res}")
                            count.main(img_in_dir, res)
                            print("\n")
                            break
                        except RuntimeError:
                            res -= 200
                else:
                    print("\twaiting for new img...", end="\r")
            sleep(5)
    except KeyboardInterrupt:
        print("\n\n\tGoodbye!\n")
        exit()
