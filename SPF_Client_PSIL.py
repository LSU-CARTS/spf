# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:14:55 2022

@author: Sam
"""
import PySimpleGUI as sg
import pandas as pd
import numpy as np
import scipy
import estimagic as em
import warnings
import matplotlib.pyplot as plt
import pyodbc
import random


#SPF function
def SPF(B1,B2,B3,b):
    terms = []
    EVs = []
    for i in range(len(Data)):    
        #Expected Value calculation
        E = Data['L'][i]*(B1*(Data['AADT'][i]**B2)/((Data['AADT'][i]**B2)+(B3**B2)))
        EVs.append(E) #adds EV to list
        #Log Likelihood calculation. 
        term = scipy.special.gammaln(b*Data['L'][i]+Data[AccType][i])-scipy.special.gammaln(b*Data['L'][i])+b*Data['L'][i]*np.log(b*Data['L'][i])+Data[AccType][i]*np.log(E)-(b*Data['L'][i]+Data[AccType][i])*np.log(b*Data['L'][i]+E)
        terms.append(term) #adds log likelihood to list
    return({'Expected Values' : EVs, 'Log Likelihoods' : terms})
    
#Function for taking the sum of the log-likelihood values 
#for a given set of parameters for the optimizer to work off.
def SumLogLik(X):
    S = np.sum(SPF(X[0],X[1],X[2],X[3])['Log Likelihoods'])
    return(S)

#Similar function for EVs for the optimizer to set the constraint.    
def SumEVs(X):
    S = np.sum(SPF(X[0],X[1],X[2],X[3])['Expected Values'])
    return(S)

#Optimizer function. Defaults to 3 starts and baseline 
#orders of magnitude to randomize the start position from.
def Multistart(starts=3,basestart=[1000,1,10**7,1]):
    x=0
    LogLik = 0
    Soln = 0
    warnings.filterwarnings("error") #treats runtime warnings as errors.
    while x < starts:
        try: #runs the block of code unless it throws an error
            #optimizes SumLogLik from a randomized fraction of the basestart
            #sets a constraint that SumEVs is equal to the sum of the accident counts column
            #sets upper and lower bounds for the parameters
            xSoln = em.maximize(criterion=SumLogLik, params=basestart*np.random.rand(4,), algorithm='scipy_slsqp', 
                               constraints={"type": "nonlinear", "func": SumEVs, "value": np.sum(Data[AccType])},
                               lower_bounds=np.array([0.001,0.001,0.001,0.001]),
                               upper_bounds=np.array([10**10,100,10**12,100]))
            #Uses the solution parameters to calculate sum log likelihood
            xLL = SumLogLik(xSoln.params)
            #Checks sum log likelihood against previous best solution
            #(or 0 for the first start) and replaces previous best if better
            if SumLogLik(xSoln.params) > LogLik or x == 0:
                LogLik = xLL
                Soln = xSoln
            x += 1 #increases index to move forward to next start
        except:
            continue #retries same start in case of a runtime warning/error
    return(Soln)

sg.theme('DarkAmber')  

#GUI asks user for highway class and accident type from dropdown menus, and for a file path to save output to.
layout = [  [sg.Text('Welcome to SPF Maker')],
            [sg.Text('Highway Class'), sg.Combo(['RURAL 1-WAY 1-LANE', 'URBAN 1-WAY 1-LANE',
                                                 'RURAL 1-WAY 2-LANE', 'URBAN 1-WAY 2-LANE',
                                                 'URBAN 1-WAY 4-LANE', 'URBAN 2-LANE INTERSTATE',
                                                 'URBAN 3-LANE INTERSTATE', 'RURAL 4-LANE INTERSTATE',
                                                 'URBAN 4-LANE INTERSTATE', 'URBAN 5-LANE INTERSTATE',
                                                 'RURAL 6-LANE INTERSTATE', 'URBAN 6-LANE INTERSTATE',
                                                 'URBAN 8-LANE INTERSTATE', 'RURAL 2-LANE',
                                                 'URBAN 2-LANE', 'RURAL 4-LANE',
                                                 'URBAN 4-LANE', 'RURAL 4-LANE DIVIDED',
                                                 'URBAN 4-LANE DIVIDED', 'URBAN 4-LANE FREEWAY',
                                                 'RURAL 2-LANE DIVIDED', 'URBAN 2-LANE DIVIDED',
                                                 'URBAN 3-LANE', 'RURAL 3-LANE DIVIDED',
                                                 'URBAN 3-LANE DIVIDED', 'URBAN 6-LANE',
                                                 'URBAN >=6-LANE DIVIDED', 'RURAL 2-LANE CONT TURN',
                                                 'URBAN 2-LANE CONT TURN', 'RURAL 4-LANE CONT TURN',
                                                 'URBAN 4-LANE CONT TURN', 'URBAN >=6-LANE CONT TURN',
                                                 'OTHER RURAL ROADS','OTHER URBAN ROADS'], readonly=True)],
            [sg.Text('Column for accident counts, eg. TotalCrashes'), sg.Combo(['TotalCrashes', 'TotalInjury', 'Fatal', 'Serious',
                                                                                'Minor', 'Possible',
                                                                                'PDO'], readonly=True)],
            [sg.Text('AADT is between'), sg.InputText(), sg.Text('and'), sg.InputText()],
            [sg.Text('Section Length is between'), sg.InputText(), sg.Text('and'), sg.InputText()],
            [sg.Text('Max sample size'), sg.InputText()],
            [sg.Text('Path for output file'), sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]



# Create the Window
window = sg.Window('SPF Maker', layout)
# Event Loop to process "events" and get the "values" of the inputs
x = 0
while x<1:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
    #Path for Excel file which the filtering references to know which code corresponds to the desired highway class.
    HwyClassPath = 'New Hwy Class Groupings.xlsx'
    
    #Assigns user inputs to appropriate variables for code.
    HwyClass = values[0]
    AccType = values[1]
    MinAADT = int(values[2])
    MaxAADT = int(values[3])
    MinL = float(values[4])
    MaxL = float(values[5])
    MaxSample = int(values[6])
    OutputCSV = values[7] 
    
    #Filters data to get 3 column dataframe with length, AADT, and accident counts for appropriate roadway and accident types.
    HwyClasses = pd.read_excel(io=HwyClassPath)
    
    
    HwyClassIND = HwyClasses[HwyClasses['HighwayClass'] == HwyClass].index[0]
    HwyClassCode = HwyClasses['HighwayClassCode'][HwyClassIND]
    
    GroupA = [3,4,5,6,7,8]
    GroupB = [10,12,'','','','']
    GroupC = [14,16,18,'','','']
    GroupD = [15,17,'','','','']
    GroupE = [21,32,34,'','','']
    GroupF = [38,40,'','','','']
    
    if HwyClassCode in GroupA:
        HwyClassGroup = GroupA
    elif HwyClassCode in GroupB:
        HwyClassGroup = GroupB
    elif HwyClassCode in GroupC:
        HwyClassGroup = GroupC
    elif HwyClassCode in GroupD:
        HwyClassGroup = GroupD
    elif HwyClassCode in GroupE:
        HwyClassGroup = GroupE
    elif HwyClassCode in GroupF:
        HwyClassGroup = GroupF
    else:
        HwyClassGroup = [HwyClassCode,'','','','','']
    
    #Path for PSI list file.
    DB_Path = 'All_PSI_2022.xlsx'

    DB_output = pd.read_excel(io=DB_Path)
    
    #Filters by highway class, converts relevant columns back into numeric values from strings, and filters out null AADT values which are recorded as 0.
    DB_filtered = DB_output[(DB_output['HwyClass'] == HwyClassGroup[0]) |
                            (DB_output['HwyClass'] == HwyClassGroup[1]) |
                            (DB_output['HwyClass'] == HwyClassGroup[2]) |
                            (DB_output['HwyClass'] == HwyClassGroup[3]) |
                            (DB_output['HwyClass'] == HwyClassGroup[4]) |
                            (DB_output['HwyClass'] == HwyClassGroup[5])]
    DB_filtered = DB_filtered[DB_filtered['AvgAADT'] >= max(1,MinAADT)].sort_values(by='AvgAADT')
    DB_filtered = DB_filtered[DB_filtered['AvgAADT'] < MaxAADT].sort_values(by='AvgAADT')
    DB_filtered = DB_filtered[DB_filtered['SegmentLength'] >= MinL]
    DB_filtered = DB_filtered[DB_filtered['SegmentLength'] < MaxL] 
    DB_filtered = DB_filtered.reset_index()
    #If user set a max sample size smaller than the full size of the sample, selects a 
    #sample of that size mostly at random but always containing the 5 largest AvgAADT values.
    if len(DB_filtered['AvgAADT']) > MaxSample:
        RandSamp = random.sample(range(len(DB_filtered['AvgAADT'])-5), MaxSample-5)
        RandSamp.sort()
        L_list = []
        AADT_list = []
        Acc_list = []
        for Index in RandSamp:
            L_list.append(DB_filtered['SegmentLength'][Index])
            AADT_list.append(DB_filtered['AvgAADT'][Index])
            if AccType == 'TotalInjury':
                Acc_list.append(DB_filtered['TotalCrashes'][Index] - DB_filtered['PDO'][Index])
            else:
                Acc_list.append(DB_filtered[AccType][Index])
        for n in range(5):
            L_list.append(DB_filtered['SegmentLength'][len(DB_filtered['AvgAADT'])-5+n])
            AADT_list.append(DB_filtered['AvgAADT'][len(DB_filtered['AvgAADT'])-5+n])
            if AccType == 'TotalInjury':
                Acc_list.append(DB_filtered['TotalCrashes'][len(DB_filtered['AvgAADT'])-5+n] - DB_filtered['PDO'][len(DB_filtered['AvgAADT'])-5+n])
            else:
                Acc_list.append(DB_filtered[AccType][len(DB_filtered['AvgAADT'])-5+n])
        Data = pd.DataFrame({'L' : L_list, 'AADT' : AADT_list, AccType : Acc_list})
    else:
        if AccType == 'TotalInjury':
            Data = pd.DataFrame({'L' : DB_filtered['SegmentLength'], 'AADT' : DB_filtered['AvgAADT'], AccType : DB_filtered['TotalCrashes'] - DB_filtered['PDO']})
        else:
            Data = pd.DataFrame({'L' : DB_filtered['SegmentLength'], 'AADT' : DB_filtered['AvgAADT'], AccType : DB_filtered[AccType]})

    y = 0.2 #y value. Represents a 5 year sample size.
    params = Multistart().params #Runs optimizer function and gets parameters
    
    #Filling out the table based on the optimized parameters
    Data['N_fit'] = SPF(params[0],params[1],params[2],params[3])['Expected Values']
    Data['Log Likelihood'] = SPF(params[0],params[1],params[2],params[3])['Log Likelihoods']
    Data['N_TL/mi/y'] = y*Data[AccType]/Data['L']
    Data['N_p/mi/y'] = y*Data['N_fit']/Data['L']
    Data['Residual'] = Data[AccType]-Data['N_fit']
    Data['Sqr Res'] = Data['Residual']**2

    CumRes = []
    CumSqrRes = []
    p80th = []
    p20th = []
    sigmas = []
    W_fs = []
    EB_freqs = []
    N_EBs = []
    for i in range(len(Data)):
        CR = np.sum(Data['Residual'][:i+1])
        CSR = np.sum(Data['Sqr Res'][:i+1])
        gi80 = scipy.stats.gamma.ppf(0.8, params[3], 0, (Data['N_p/mi/y'][i]/params[3]))
        gi20 = scipy.stats.gamma.ppf(0.2, params[3], 0, (Data['N_p/mi/y'][i]/params[3]))
        CumRes.append(CR)
        CumSqrRes.append(CSR)
        p80th.append(gi80)
        p20th.append(gi20)
        sigma = np.sqrt(CSR)*np.sqrt(1-(CSR/np.sum(Data['Sqr Res'])))
        sigmas.append(sigma)
        W_f = 1/(1+(Data['N_fit'][i]/(params[3]*Data['L'][i])))
        W_fs.append(W_f)
        EB_freq = W_f*Data['N_fit'][i]+(1-W_f)*Data[AccType][i]
        EB_freqs.append(EB_freq)
        N_EB = y*EB_freq/Data['L'][i]
        N_EBs.append(N_EB)
        
    Data['Cum Res'] = CumRes
    Data['Cum Sqr Res'] = CumSqrRes
    Data['80th/m/y'] = p80th
    Data['20th/m/y'] = p20th
    Data['sigma(i)'] = np.sqrt(Data['Cum Sqr Res'])
    Data['sigma'] = sigmas
    Data['W_f'] = W_fs
    Data['EB Freq'] = EB_freqs
    Data['N_EB/mi/y'] = N_EBs

    #LOSS function
    LOSS = []
    for i in range(len(Data)):
        if Data['N_EB/mi/y'][i] > Data['80th/m/y'][i]:
            LOSS.append('LOSS 4')
        elif Data['N_EB/mi/y'][i] > Data['N_p/mi/y'][i]:
            LOSS.append('LOSS 3')
        elif Data['N_EB/mi/y'][i] > Data['20th/m/y'][i]:
            LOSS.append('LOSS 2')
        else:
            LOSS.append('LOSS 1')
    Data['LOSS'] = LOSS

    #makes figure for scatterplot
    plt.figure()
    plt.scatter(Data['AADT'], Data['N_TL/mi/y'], label='N_TL/mi/y')
    plt.scatter(Data['AADT'], Data['N_EB/mi/y'], label='N_EB/mi/y')
    plt.plot(Data['AADT'], Data['N_p/mi/y'], 'k-', label='N_p/mi/y')
    plt.plot(Data['AADT'], Data['80th/m/y'], 'r--', label='80th/m/y')
    plt.plot(Data['AADT'], Data['20th/m/y'], 'g--', label='20th/m/y')
    plt.legend()
    plt.savefig('SPFscatter.png')

    #makes figure for cureplot
    plt.figure()
    plt.scatter(Data['AADT'], Data['Cum Res'], label='Cum Res')
    plt.plot(Data['AADT'], 2*Data['sigma'], 'r--', label='+2\u03C3')
    plt.plot(Data['AADT'], -2*Data['sigma'], 'g--', label='-2\u03C3')
    plt.legend()
    plt.savefig('SPF_CURE.png')

    #dataframe for parameters and y value to output into Excel
    paramframe = pd.DataFrame({'y' : [y], 'B1' : params[0:1], 'B2' : params[1:2],
                               'B3' : params[2:3], 'b' : params[3:4]})

    #writes parameters, filled out data table, and figures to Excel file.
    writer = pd.ExcelWriter(OutputCSV, engine='xlsxwriter')

    paramframe.to_excel(writer)
    Data.to_excel(writer,startrow=6)
    writer.sheets['Sheet1'].insert_image('V8', 'SPFscatter.png')
    writer.sheets['Sheet1'].insert_image('V30', 'SPF_CURE.png')

    writer.save()
    print('Created '+OutputCSV)
    x += 1
    
window.close()