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


    filename = "data/features_numeric_test"+str(num)+".txt"


    file_data = open(filename).readlines()

    # outname1 = "data/features_numeric_tr"+str(num)+".txt"
    # outname2 = "data/features_numeric_te"+str(num)+".txt"


    # f_out1 = open("data/features_numeric_tr4.txt","w+")
    # f_out2 = open("data/features_numeric_te4.txt","w+")

    outname = "data/features_numeric_test"+str(num)+"_binary.txt"
    f_out2 = open(outname,"w+")

    count = 0

    for line in file_data:
        line_list = line.split()
        result = ''


        # tag
        result += line_list[0]+' '



        # checking_status
        if line_list[1] == '1':
            result += '0 0 0 1 '
        elif line_list[1] == '2':
            result += '0 0 1 0 '
        elif line_list[1] == '3':
            result += '0 1 0 0 '
        elif line_list[1] == '4':
            result += '1 0 0 0 '


        # duration
        if int(line_list[2]) < 10:
            result += '0 0 0 '
        elif int(line_list[2]) >= 10 and int(line_list[2]) < 20:
            result += '0 0 1 '
        elif int(line_list[2]) >= 20 and int(line_list[2]) < 30:
            result += '0 1 1 '
        elif int(line_list[2]) >= 30:
            result += '1 1 1 '

        # credit_history
        if line_list[3] == '0':
            result += '1 0 0 0 0 '
        elif line_list[3] == '1':
            result += '0 1 0 0 0 '
        elif line_list[3] == '2':
            result += '0 0 1 0 0 '
        elif line_list[3] == '3':
            result += '0 0 0 1 0 '
        elif line_list[3] == '4':
            result += '0 0 0 0 1 '

        # purpose

        # credit_amount
        if int(line_list[4]) < 20:
            result += '0 0 0 '
        elif int(line_list[4]) >= 20 and int(line_list[4]) < 40:
            result += '0 0 1 '
        elif int(line_list[4]) >= 40 and int(line_list[4]) < 60:
            result += '0 1 1 '
        elif int(line_list[4]) >= 60:
            result += '1 1 1 '


        # savings_status
        if line_list[5] == '1':
            result += '0 0 0 0 '
        elif line_list[5] == '2':
            result += '0 0 1 0 '
        elif line_list[5] == '3':
            result += '0 1 1 0 '
        elif line_list[5] == '4':
            result += '1 1 1 0 '
        elif line_list[5] == '5':
            result += '0 0 0 1 '


        # employment
        if line_list[6] == '1':
            result += '0 0 0 1 '
        elif line_list[6] == '2':
            result += '0 0 0 0 '
        elif line_list[6] == '3':
            result += '0 0 1 0 '
        elif line_list[6] == '4':
            result += '0 1 1 0 '
        elif line_list[6] == '5':
            result += '1 1 1 0 '

        # installment_commitment

        #personal_status
        if line_list[7] == '1':
            result += '1 0 0 0 0 '
        elif line_list[7] == '2':
            result += '0 1 0 0 0 '
        elif line_list[7] == '3':
            result += '0 0 1 0 0 '
        elif line_list[7] == '4':
            result += '0 0 0 1 0 '
        elif line_list[7] == '5':
            result += '0 0 0 0 1 '

        # other_parties

        #residence_since
        if int(line_list[8]) <= 2:
            result += '0 '
        elif int(line_list[8]) > 2:
            result += '1 '

        #property_magnitude
        if line_list[9] == '1':
            result += '1 0 0 0 '
        elif line_list[9] == '2':
            result += '0 1 0 0 '
        elif line_list[9] == '3':
            result += '0 0 1 0 '
        elif line_list[9] == '4':
            result += '0 0 0 1 '

        #age
        if int(line_list[10]) < 20:
            result += '0 0 0 '
        elif int(line_list[10]) >= 20 and int(line_list[10]) < 40:
            result += '0 0 1 '
        elif int(line_list[10]) >= 40 and int(line_list[10]) < 60:
            result += '0 1 1 '
        elif int(line_list[10]) >= 60:
            result += '1 1 1 '


        #other_payment_plans
        if line_list[11] == '3':
            result += '0 '
        else:
            result += '1 '



        # housing

        # existing_credits
        if line_list[12] == '1':
            result += '1 0 0 '
        elif line_list[12] == '2':
            result += '0 1 0 '
        elif int(line_list[12]) >= 3:
            result += '0 0 1 '

        

        # job

        # num_dependent
        if int(line_list[13]) < 2:
            result += '0 '
        else:
            result += '1 '

        # own_telephone 
        if line_list[14] == '1':
            result += '0 '
        else:
            result += '1 '


        # foreign_worker
        if line_list[15] == '1':
            result += '0 '
        else:
            result += '1 '


        #purpose
        # %Attribute 4: 1 0-->A40(car(new))  0 1-->A41(car(used))  0 0-->rest catagories
        result += line_list[16] + ' ' + line_list[17] + ' '

        #other_parties
        # %Attribute 10: 1 0-->A101(none)  0 1-->A102  0 0-->A103
        result += line_list[18] + ' ' + line_list[19] + ' '


        #housing
        # %Attribute 15: 1 0-->A151  0 1-->A152 0 0-->A153
        result += line_list[20] + ' ' + line_list[21] + ' '

        #job
        # %Attribute 17: 1 0 0-->A171  0 1 0-->A172  0 0 1-->A174  0 0 0-->A174
        result += line_list[22] + ' ' + line_list[23] + ' ' + line_list[24] + ' '


        result += '\n'

        f_out2.write(result)




















    






            


