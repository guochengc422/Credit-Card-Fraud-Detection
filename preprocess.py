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


    filename = "data/features_numeric_train"+str(num)+"_binary.txt"
    # filename2 = "data/features_numeric_test"+str(num)+".txt"


    file_data = open(filename).readlines()
    # file_data2 = open(filename2).readlines()

    outname1 = "data/features_numeric_tr"+str(num)+".txt"
    outname2 = "data/features_numeric_te"+str(num)+".txt"


    f_out1 = open(outname1,"w+")
    f_out2 = open(outname2,"w+")

    # outname = "data/features_numeric.txt"
    # f_out2 = open(outname,"w+")

    count = 0

    for line in file_data:

        if count < 600:
            f_out1.write(line)
        else:
            f_out2.write(line)

        count += 1

    


    






            


