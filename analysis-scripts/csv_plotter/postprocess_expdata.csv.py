__author__ = 'anilpa'

import csv
with open('/Users/anilpa/GitHub/OnlineRegression/data/reporting/comparison_table_online.csv', 'rb') as csvfile:
    exp_data = csv.reader(csvfile, delimiter='\t')

    with open('/Users/anilpa/GitHub/OnlineRegression/data/reporting/comparison_table_online_processed_nofiltered.csv', 'wb') as csvfile2:
        processed_exp_data = csv.writer(csvfile2, delimiter=',')

        processed_exp_data.writerow(["Variant", "Algorithm", "IsBatch", "IsParametric", "IsWindowed", "IsMapped", "WindowSize", "ForgettingFactor",
                           "Test", "InputDimensionality", "StreamSize", "IsSynthetic", "Discontinuity", "IsStatic", "InputScale", "NoiseVariance", "DriftType",
                           "SMSE", "SMSE_ST", "RMSE", "RMSE_ST", "AIW", "SAIW", "ICR",
                            "APT", "AUT", "ATT", "HPT", "HUT", "HTT", "TPT", "TUT", "TTT", "TT", "ATPT", "UC", "TC"])

        ctr = 0
        for row in exp_data:
            # Extracting algorithm details

            ctr = ctr + 1

            if ctr == 1:
                continue

            # print row

            # if float(row[2]) > 10.0 or float(row[3]) > 1.0: # invalid point!
            #     continue

            # if float(row[7]) > 1.0: # invalid point!
            #     continue

            variant = row[0]

            # only applicaple to windowed learners
            windowed = False
            window_size = -1

            # only applicaple to learners with forgetting mechanism
            forgetting = False
            forgetting_factor = -1

            # only applicaple to batch learners
            batch = False
            training_size = -1

            # only applicaple to GP variants
            meanfunc = "NA"

            if "WS" in variant:
                windowed = True
                window_size = int(variant.split("WS")[1])
            elif "FR" in variant:
                forgetting = True
                forgetting_factor = int(variant.split("FR")[1])/100.0
                row[3] = row[2]
                row[5] = row[4]
            elif "Batch" in variant:
                batch = True
                row[3] = row[2]
                row[5] = row[4]

            algorithm = "NA"
            mapped = False

            if "GP" in variant:
                parametric = False
                algorithm = "GPRegression"
                index = variant.find("Mean")
                temp = variant[index-3:index]
                if temp == "OLS":
                    meanfunc = "OLS"
                elif temp == "Avg":
                    meanfunc = "AVG"
                else:
                    meanfunc = "Zero"
            elif "KernelRegression" in variant:
                parametric = False
                algorithm = "KernelRegression"
            else:
                parametric = True

                if "Mapped" in variant:
                    mapped = True
                else:
                    mapped = False

                if "MAP" in variant:
                    algorithm = "BayesianMAP"
                elif "MLE" in variant:
                    algorithm = "BayesianMLE"

            # Extracting test details
            test = row[1]

            synthetic = False
            # Only applicable to synthetic datasets! otherwise false
            discontinuity = False
            static = False
            input_scale = -1
            noise_var = -1
            drift_type = "NA"

            temp = test.split('_')

            size = temp[3]
            dimension = int(temp[4])

            if "SYNTH" in test:
                synthetic = True

                if temp[1] == 'D':
                    discontinuity = True

                if temp[2] == 'NCD':
                    static = True

                input_scale = int(temp[5])
                noise_var = int(temp[6])
                drift_type = temp[7]

            # Postporcessing Experiment data

            smse = float(row[2])
            smse_t = float(row[3])
            rmse = float(row[4])
            rmse_t = float(row[5])
            aiw = float(row[6])
            saiw = float(row[7])
            icr = float(row[8])/100.0
            apt = float(row[9])
            aut = float(row[10])
            att = float(row[11])
            hpt = float(row[12])
            hut = float(row[13])
            htt = float(row[14])
            tpt = float(row[15])
            tut = float(row[16])
            ttt = float(row[17])
            tt = float(row[18])
            atpt = float(row[19])
            uc = int(row[20])
            tc = int(row[21])

            # if smse_t > smse:
            #     smse_t = smse
            # if rmse_t > rmse:
            #     rmse_t = rmse

            # if saiw < 0.01:
            #     continue


            # print smse, smse_t, rmse, rmse_t
            processed_exp_data.writerow([variant, algorithm, batch, parametric, windowed, mapped, window_size, forgetting_factor, test, dimension, size, synthetic, discontinuity, static, input_scale, noise_var, drift_type, smse, smse_t, rmse, rmse_t, aiw, saiw, icr,
                            apt, aut, att, hpt, hut, htt, tpt, tut, ttt, tt, atpt, uc, tc])










