#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      GuoCheng
#
# Created:     19/06/2015
# Copyright:   (c) GuoCheng 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import fileinput

if __name__ == '__main__':


    num = 4


    filename = "data/features_numeric.txt"

    file_data = open(filename).readlines()


    count_all = 0

    count_correct = 0

    count_detected = 0

    for line in file_data:

        linelist = line.split()

        tag = linelist[0]

        if tag == '1':
            count_all += 1

        if int(linelist[2]) >= 20 and int(linelist[1]) == 1 and (linelist[18] == '1' or linelist[19] == '1') and linelist[20] == '1':
            count_detected += 1

        if tag == '1' and int(linelist[2]) >= 20 and int(linelist[1]) == 1 and (linelist[18] == '1' or linelist[19] == '1') and linelist[20] == '1':
            count_correct += 1

    print count_all
    print count_detected
    print count_correct

    


    






            


