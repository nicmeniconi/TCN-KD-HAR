import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.signalProcessing as signalutil
import utils.fileProcessing as fileutil


lower_activities = ["A01","A02","A03","A04"]
upper_activities = ["A05","A06","A07","A08","A09","A10","A11","A12","A13"]
dataset_activities = lower_activities + upper_activities

motsignals_range = {
    'knee_angle_r':(-20,120),
    'knee_angle_l':(-20,120),
    'arm_flex_r':(-90,180),
    'elbow_flex_r':(-10,180),
    'arm_flex_l':(-90,190),
    'elbow_flex_l':(-10,190),
}

def getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity):
    torso = dfcsv[['torso_x', 'torso_y', 'torso_z']].to_numpy()
    neck = dfcsv[['neck_x', 'neck_y', 'neck_z']].to_numpy()
    rshoulder = dfcsv[['right_shoulder_x','right_shoulder_y', 'right_shoulder_z']].to_numpy()
    lshoulder = dfcsv[['left_shoulder_x','left_shoulder_y', 'left_shoulder_z']].to_numpy()
    relbow = dfcsv[['right_elbow_x', 'right_elbow_y', 'right_elbow_z']].to_numpy()
    lelbow = dfcsv[['left_elbow_x', 'left_elbow_y', 'left_elbow_z']].to_numpy()
    rwrist = dfcsv[['right_wrist_x', 'right_wrist_y', 'right_wrist_z']].to_numpy()
    lwrist = dfcsv[['left_wrist_x', 'left_wrist_y', 'left_wrist_z']].to_numpy()
    rhip = dfcsv[['right_hip_x', 'right_hip_y', 'right_hip_z']].to_numpy()
    lhip = dfcsv[['left_hip_x', 'left_hip_y', 'left_hip_z']].to_numpy()
    rknee = dfcsv[['right_knee_x', 'right_knee_y', 'right_knee_z']].to_numpy()
    lknee = dfcsv[['left_knee_x', 'left_knee_y', 'left_knee_z']].to_numpy()
    rankle = dfcsv[['right_ankle_x', 'right_ankle_y', 'right_ankle_z']].to_numpy()
    lankle = dfcsv[['left_ankle_x', 'left_ankle_y', 'left_ankle_z']].to_numpy()
    # LOWER ACTIVITIES  
    if activity in lower_activities:
        if activity in ["A02", "A04"]:
            jointMot = "knee_angle_r"
            bone1 = rhip
            bone2 = rknee
            bone3 = rknee
            bone4 = rankle
        elif activity in ["A01", "A03"]:
            jointMot = "knee_angle_l"
            bone1 = lhip
            bone2 = lknee
            bone3 = lknee
            bone4 = lankle
    # UPPER ACTIVITIES 
    elif activity in upper_activities:
        if activity in ["A05","A09"]: 
            jointMot = "elbow_flex_r"        
            bone1 = rshoulder
            bone2 = relbow
            bone3 = relbow
            bone4 = rwrist
        elif activity in ["A06"]: 
            jointMot = "elbow_flex_l"          
            bone1 = lshoulder
            bone2 = lelbow
            bone3 = lelbow
            bone4 = lwrist
        elif activity in ["A07","A10","A11","A13"]:
            jointMot = "arm_flex_r" #RIGHT SHOULDER
            bone1 = rshoulder
            bone2 = relbow
            bone3 = neck
            bone4 = torso
        elif activity in ["A08","A12"]:
            jointMot = "arm_flex_l" #LEFT SHOULDER
            bone1 = lshoulder
            bone2 = lelbow
            bone3 = neck
            bone4 = torso
        else:
            jointMot = "None"
    
    bonesCSV = [bone1,bone2,bone3,bone4]
    return jointMot,bonesCSV

def getSamplesToSynchronize(jointangle_imu,jointangle_video,seconds=10,hz=30,maxshift=15,fitlength=None):
    samples = min(len(jointangle_imu),len(jointangle_video))
    if not fitlength:
        fitlength = int(samples)-2*hz

    jointangle_imus_subarray=jointangle_imu[0:fitlength]
    jointangle_video_subarray = jointangle_video[0:fitlength]
    rmse_original = signalutil.calcRMSE(jointangle_imus_subarray,jointangle_video_subarray)
    #print('RMSE before shifting: ', rmse_original)

    remove_samples_video = 0
    minimum_rmse_video = rmse_original
    for n in range(0,maxshift):
        jointangle_imus_subarray=jointangle_imu[0:fitlength]
        jointangle_video_subarray = jointangle_video[n:fitlength+n]
        rmse_video = signalutil.calcRMSE(jointangle_imus_subarray,jointangle_video_subarray)
        if rmse_video < minimum_rmse_video:
            #print('NEW MINIMUM RMSE (shifting video): ',rmse_video)
            minimum_rmse_video = rmse_video
            remove_samples_video = n

    # Desplazar la señal de IMUs a la izquierda
    remove_samples_imu = 0
    minimum_rmse_imu = rmse_original
    for n in range(0,maxshift):
        jointangle_imus_subarray=jointangle_imu[n:fitlength+n]
        jointangle_video_subarray = jointangle_video[0:fitlength]
        rmse_imu = signalutil.calcRMSE(jointangle_imus_subarray,jointangle_video_subarray)
        if rmse_imu < minimum_rmse_imu:
            #print('NEW MINIMUM RMSE (shifting imus): ',rmse_imu)
            minimum_rmse_imu = rmse_imu
            remove_samples_imu = n

    bool_cut_imu=False
    bool_cut_video=False
    #Decide what to cut depending on rmse
    if minimum_rmse_video < minimum_rmse_imu:
        #print("Cut VIDEO signal {:} for rmse {:.4f}".format(remove_samples_video,minimum_rmse_video)+ " (instead of IMU signal {:} for rmse {:.4f})".format(remove_samples_imu,minimum_rmse_imu))
        bool_cut_video=True
    elif minimum_rmse_imu < minimum_rmse_video:
        #Cut IMU signal
        bool_cut_imu=True
        #print("Cut IMUs signal {:} (x2 in text file) for rmse {:.4f}".format(remove_samples_imu,minimum_rmse_imu)+ " (instead of VIDEO signal {:} for rmse {:.4f})".format(remove_samples_video,minimum_rmse_video))
    # else:
    #     print("Minimum_rmse NOT FOUND")

    return rmse_original, remove_samples_imu,minimum_rmse_imu,bool_cut_imu,remove_samples_video,minimum_rmse_video,bool_cut_video

def SynchronizeAndCutSignals(jointangle_imus,jointangle_video,remove_samples_imu,remove_samples_video,max_length=None):
    
    #align signals
    jointangle_imus_new=jointangle_imus[remove_samples_imu:]
    jointangle_video_new=jointangle_video[remove_samples_video:]

    #cutsignals
    imu_length = len(jointangle_imus_new)
    video_length = len(jointangle_video_new)
    if max_length is None:
        max_length = min(imu_length,video_length)
    
    #print(max_length)
    jointangle_imus_final=jointangle_imus_new[:max_length]
    jointangle_video_final=jointangle_video_new[:max_length]
    mse = 1000
    rmse = 100
    try:
        rmse = signalutil.calcRMSE(jointangle_imus_final,jointangle_video_final)
    except:
        #print("Mse {:}, rmse: {:}".format(mse,rmse))
        rmse = 1000
    return jointangle_imus_final,jointangle_video_final,rmse

def addFramesShiftToCSVLog(csvlog,subject,activity,trial,filename,type,cutframes,origrmse,theormse):
    
    if os.path.exists(csvlog):
        dfchanges = pd.read_csv(csvlog)
    else:
        dfcolumns = [
                'Subject',
                'Activity',
                'Trial',
                'File',
                'Type',
                'CutFrames',
                'OrigRmse',
                'TheoRmse'
            ]
        dfchanges = pd.DataFrame(columns=dfcolumns)
    
    dictChanges = {
        'Subject':[subject],
        'Activity':[activity],
        'Trial':[trial],
        'File':[filename],
        'Type':[type],
        'CutFrames':[cutframes],
        'OrigRmse':[origrmse],
        'TheoRmse':[theormse]
        }
    entry=pd.DataFrame.from_dict(dictChanges)
    dfchanges = pd.concat([dfchanges,entry],ignore_index=True)
    dfchanges.to_csv(csvlog,index=False,mode='w+')

def plotFramesShiftToSyncrhonizeAllSubjectsOneActivity(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    ncols = 5
    nrows = len(subjects)

    csvlogfile = os.path.join(outpath,csvlog)

    fig,axes=plt.subplots(nrows,ncols,figsize=(ncols*6,nrows*3))
    plt.rc('axes', titlesize=15)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # 1) Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject+"_"+activity+"_"+trial
            motfilename = 'ik_'+motsubjacttrial+".mot"
            inpathmotfull = os.path.join(inpath,subject,motfilename)
            if not os.path.exists(inpathmotfull):
                continue
            else:
                folder = os.path.join(inpath,subject)
                #print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue

        jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
        jointangle_video_inter = signalutil.fill_nan(jointangle_video)
        jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        # 3) Smooth both signals
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
        jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        
        # 4) Compute RMSE of signals
        SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
        rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
       
        # 5) Center signals in mean for better synchronization
        jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
        jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
        rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
      
        # 6) Shift and cut signals to find ideal synchronization 
        rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
        remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                            jointangle_imus_centered,
                                                            jointangle_video_centered,
                                                            maxshift=MAX_SYNC_OVERLAP,
                                                            fitlength=RMSE_SAMPLES)
        if bool_cut_imu:
            rm_samples_imu_minrmse=remove_samples_imu
            rm_samples_video_minrmse=0    
        elif bool_cut_video:
            rm_samples_imu_minrmse=0
            rm_samples_video_minrmse=remove_samples_video
        else:
            rm_samples_imu_minrmse=0
            rm_samples_video_minrmse=0

        if FINAL_LENGTH is None:
            print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
            print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
            FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
            print(FINAL_LENGTH)
        jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                            jointangle_imus_centered,
                                                            jointangle_video_centered,
                                                            rm_samples_imu_minrmse,
                                                            rm_samples_video_minrmse,
                                                            max_length=FINAL_LENGTH)

        #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

        # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
        if remove_samples_imu>0:
            imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
            
            imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)
        
        if remove_samples_video>0:
            videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)

            videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

        rmse_list.append(rmse_shift)
        yaxis0=motsignals_range[jointMot][0]
        yaxis1=motsignals_range[jointMot][1]
        
        for j in range(ncols):
            if j == 0:
                color = 'r'
                imujoint=jointMot
                X=np.arange(0,jointangle_imus.shape[0])
                axes[i][j].set_ylim([yaxis0, yaxis1])
                axes[i][j].plot(X,jointangle_imus,color)
                axes[i][j].set_title(motsubjacttrial+'.mot' + ' (Angle: '+str(imujoint)+')')
                axes[i][j].set_xlabel("")
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (50 Hz)")
            if j==1:
                color = 'b'
                X=np.arange(0,jointangle_video_cutfilt.shape[0])
                axes[i][j].set_ylim([yaxis0, yaxis1])
                axes[i][j].plot(X,jointangle_video_cutfilt,color)
                axes[i][j].set_title(motsubjacttrial+'.csv' + " (Angle: estimated)")
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
            if j==2:
                X=np.arange(0,RMSE_SAMPLES)
                axes[i][j].set_ylim([yaxis0, yaxis1])
                axes[i][j].plot(X,jointangle_imus_cutfilt[:RMSE_SAMPLES],color='r',label='imus')
                axes[i][j].plot(X,jointangle_video_cutfilt[:RMSE_SAMPLES],color='b',label='video')
                axes[i][j].set_title("SMOOTHED, RMSE: {:.2f}".format(rmse_smooth))
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
                axes[i][j].legend()
            if j==3:
                X=np.arange(0,RMSE_SAMPLES)
                axes[i][j].set_ylim([-(yaxis1-yaxis0)/2, (yaxis1-yaxis0)/2])                
                axes[i][j].plot(X,jointangle_imus_centered[:RMSE_SAMPLES],color='r',label='imus')
                axes[i][j].plot(X,jointangle_video_centered[:RMSE_SAMPLES],color='b',label='video')
                axes[i][j].set_title("MEAN REMOVAL, RMSE: {:.2f}".format(rmse_centered))
                axes[i][j].set_xlabel("")
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
                axes[i][j].legend()

            if j==4:
                X=np.arange(0,FINAL_LENGTH)
                axes[i][j].set_ylim([-(yaxis1-yaxis0)/2, (yaxis1-yaxis0)/2])                
                axes[i][j].plot(X,jointangle_imus_shift[:FINAL_LENGTH],color='r',label='imus')
                axes[i][j].plot(X,jointangle_video_shift[:FINAL_LENGTH],color='b',label='video')
                axes[i][j].set_title("SHIFTED, RMSE: {:.2f} ".format(rmse_shift)+" (cut imu:"+str(rm_samples_imu_minrmse)+", cut vid:"+str(rm_samples_video_minrmse)+")")
                axes[i][j].set_xlabel("")
                axes[i][j].set_ylabel("Degrees")
                axes[i][j].set_xlabel("Samples (30 Hz)")
                axes[i][j].legend()



    title="Activity "+activity+": "+activity_legend+ " (one subject per row)"
    plt.suptitle(title,fontsize=18, verticalalignment='top',y=1.0)
    plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
    if outputfilename:
        plt.savefig(os.path.join(outpath,outputfilename+'.svg'),format='svg')
        plt.savefig(os.path.join(outpath,outputfilename+'.pdf'),format='pdf')
        # #plt.savefig(os.path.join(outpath,outputfilename+'.png'),format='png',dpi=600)
    plt.show()
    return rmse_list

def FramesShift(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):

    csvlogfile = os.path.join(outpath,csvlog)

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # 1) Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject+"_"+activity+"_"+trial
            motfilename = 'ik_'+motsubjacttrial+".mot"
            inpathmotfull = os.path.join(inpath,subject,motfilename)
            if not os.path.exists(inpathmotfull):
                # print(f'{inpathmotfull} does not exist')
                continue
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue

        jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
        jointangle_video_inter = signalutil.fill_nan(jointangle_video)
        jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        # 3) Smooth both signals
        jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
        jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        
        # 4) Compute RMSE of signals
        SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
        rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
       
        # 5) Center signals in mean for better synchronization
        jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
        jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
        rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
      
        # 6) Shift and cut signals to find ideal synchronization 
        rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
        remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                            jointangle_imus_centered,
                                                            jointangle_video_centered,
                                                            maxshift=MAX_SYNC_OVERLAP,
                                                            fitlength=RMSE_SAMPLES)
        if bool_cut_imu:
            rm_samples_imu_minrmse=remove_samples_imu
            rm_samples_video_minrmse=0    
        elif bool_cut_video:
            rm_samples_imu_minrmse=0
            rm_samples_video_minrmse=remove_samples_video
        else:
            rm_samples_imu_minrmse=0
            rm_samples_video_minrmse=0

        if FINAL_LENGTH is None:
            print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
            print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
            FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
            print(FINAL_LENGTH)
        jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                            jointangle_imus_centered,
                                                            jointangle_video_centered,
                                                            rm_samples_imu_minrmse,
                                                            rm_samples_video_minrmse,
                                                            max_length=FINAL_LENGTH)

        #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

        # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
        if remove_samples_imu>0:
            imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
            
            imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)
        
        if remove_samples_video>0:
            videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)

            videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

        rmse_list.append(rmse_shift)

    return rmse_list

import csv
import os

def log_failed_sync(csv_logfile, subject, activity, trial, file, reason):
    file_exists = os.path.exists(csv_logfile)
    
    with open(csv_logfile, 'a', newline='') as csvfile:
        fieldnames = ['Subject', 'Activity', 'Trial', 'File', 'Reason', 'CutFrames']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Subject': subject,
            'Activity': activity,
            'Trial': trial,
            'File': file,
            'Reason': reason,
        })

def FramesShift2(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations

    rmse_list = []
    for i, subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # 1) Load/compute IMU and video joint angle signals
        # Try to open the first existing trial for that subject

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            
            if not os.path.exists(inpathmotfull):
                # Log file not found
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue
            else:
                folder = os.path.join(inpath, subject)
                # Load data
                dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
                if dfmot is None or dfcsv is None:
                    log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Data not found or missing content")
                    continue
                break  # Limit to the first existing trial

        if dfmot is None or dfcsv is None:
            print(f"No valid data found for {subject} - {activity}")
            log_failed_sync(failed_csv_logfile, subject, activity, "T01-T05", "N/A", "Data not found in all trials")
            continue

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue
        
        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0    
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)
            
            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)
        
        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    return rmse_list

def FramesShift3(csvlog, inpath, outpath, subjects, activity, activity_legend, outputfilename=None, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_sync_logfile = os.path.join(outpath, 'failedSync.csv')
    
    rmse_list = []
    for i, subject in enumerate(subjects):
        dfmot = None
        dfcsv = None

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)

            if not os.path.exists(inpathmotfull):
                log_failed_sync(failed_sync_logfile, subject, activity, trial, inpathmotfull, 'File not found')
                continue

            folder = os.path.join(inpath, subject)
            dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)

            if dfmot is None or dfcsv is None:
                log_failed_sync(failed_sync_logfile, subject, activity, trial, 'MOT or CSV data missing', 'Missing Data')
                continue
            
            try:
                # Processing and synchronization logic
                jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
                jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
                jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)

                # Downsample and smooth signals
                jointangle_video_inter = signalutil.fill_nan(jointangle_video)
                jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
                jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
                jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)

                # RMSE calculation
                SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
                rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
                
                # Check if RMSE is too high, log failure if so
                if rmse_smooth > 0.6:  # Define a threshold RMSE value
                    log_failed_sync(failed_sync_logfile, subject, activity, trial, 'High RMSE', f'RMSE too high: {rmse_smooth}')
                    continue
                
                # Proceed with synchronization
                jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
                jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
                
                rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                    jointangle_imus_centered, jointangle_video_centered, maxshift=MAX_SYNC_OVERLAP, fitlength=RMSE_SAMPLES)

                if bool_cut_imu or bool_cut_video:
                    # Log if the synchronization is not ideal or the data was heavily cut
                    log_failed_sync(failed_sync_logfile, subject, activity, trial, 'Sync Issue', 'Cut too many frames')

                # Log successful synchronization or other issues
                rmse_list.append(rmse_smooth)

            except Exception as e:
                log_failed_sync(failed_sync_logfile, subject, activity, trial, 'Processing error', str(e))

    return rmse_list

# def allfiles_sync(csv_logfile, subject, activity, trial, file, note='null'):
#     file_exists = os.path.exists(csv_logfile)
    
#     with open(csv_logfile, 'a', newline='') as csvfile:
#         fieldnames = ['Subject', 'Activity', 'Trial', 'File', 'Note']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         # Write header if file is new
#         if not file_exists:
#             writer.writeheader()
        
#         writer.writerow({
#             'Subject': subject,
#             'Activity': activity,
#             'Trial': trial,
#             'File': file,
#             'Note': note
#         })

import os
import csv

def allfiles_sync(csv_logfile, subject, activity, trial, file, note='null'):
    # Check if the file exists before appending
    file_exists = os.path.exists(csv_logfile)
    
    try:
        with open(csv_logfile, 'a', newline='') as csvfile:
            fieldnames = ['Subject', 'Activity', 'Trial', 'File', 'Note']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Log the processed file
            writer.writerow({
                'Subject': subject,
                'Activity': activity,
                'Trial': trial,
                'File': file,
                'Note': note
            })
        print(f"Logged: {subject}, {activity}, {trial}, {file}")
    except Exception as e:
        print(f"Error writing to allfiles.csv: {e}")


def FramesShift4(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath

    rmse_list = []
    for i, subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # 1) Load/compute IMU and video joint angle signals
        # Try to open the first existing trial for that subject

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            
            if not os.path.exists(inpathmotfull):
                # Log file not found
                # log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue
            else:
                folder = os.path.join(inpath, subject)
                # Load data
                dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
                if dfmot is None or dfcsv is None:
                    log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Data not found or missing content")
                    continue
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull)

                break  # Limit to the first existing trial

        if dfmot is None or dfcsv is None:
            print(f"No valid data found for {subject} - {activity}")
            log_failed_sync(failed_csv_logfile, subject, activity, "T01-T05", "N/A", "Data not found in all trials")
            continue

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue
        
        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0    
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)
            
            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)
        
        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    return rmse_list

def FramesShift5(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath

    rmse_list = []
    for i, subject in enumerate(subjects):
        dfmot = None
        dfcsv = None

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            
            # Log if file not found
            if not os.path.exists(inpathmotfull):
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue  # Continue to the next trial

            folder = os.path.join(inpath, subject)
            # Load data
            dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
            if dfmot is None or dfcsv is None:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Data not found or missing content")
                continue
            allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull)

        # If all trials failed for this subject, log it
        if dfmot is None or dfcsv is None:
            log_failed_sync(failed_csv_logfile, subject, activity, "T01-T05", "N/A", "Data not found in all trials")
            continue

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue
        
        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0    
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)
            
            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)
        
        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    return rmse_list

def FramesShift6(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath

    rmse_list = []
    for i, subject in enumerate(subjects):
        dfmot = None
        dfcsv = None

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            
            # Log if file not found
            if not os.path.exists(inpathmotfull):
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue  # Continue to the next trial

            folder = os.path.join(inpath, subject)
            # Load data
            dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
            if dfmot is None or dfcsv is None:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Data not found or missing content")
                continue
            allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')

        # If all trials failed for this subject, log it
        if dfmot is None or dfcsv is None:
            log_failed_sync(failed_csv_logfile, subject, activity, "T01-T05", "N/A", "Data not found in all trials")
            continue

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue
        
        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0    
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)
            if rmse_shift == 0:
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'rmse=0')
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)
            
            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)
        
        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    return rmse_list



def FramesShift7(csvlog, inpath, outpath, subjects, activity, activity_legend, outputfilename=None, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath

    rmse_list = []
    for i, subject in enumerate(subjects):
        dfmot = None
        dfcsv = None

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            
            # Log if file not found
            if not os.path.exists(inpathmotfull):
                # log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue  # Continue to the next trial

            folder = os.path.join(inpath, subject)
            # Load data
            dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
            if dfmot is None or dfcsv is None:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Data not found or missing content")
                continue
            allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')

        # If all trials failed for this subject, log it
        if dfmot is None or dfcsv is None:
            log_failed_sync(failed_csv_logfile, subject, activity, "T01-T05", "N/A", "Data not found in all trials")
            continue

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
      
        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue
        
        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0    
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)
            if rmse_shift == 0:
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'rmse=0')
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)
            
            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)
        
        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    # === Add Validation for File Length Consistency ===
    df_info = pd.read_csv(os.path.join(outpath, 'infoToSync.csv'))
    try:
        df_fail = pd.read_csv(os.path.join(outpath, 'failedSync.csv'))
    except:
        df_fail = pd.DataFrame()
    df_all = pd.read_csv(os.path.join(outpath, 'allfiles.csv'))

    total_synced_failed = len(df_info) + len(df_fail)
    if len(df_all) != total_synced_failed:
        raise ValueError(f"File count mismatch: {len(df_all)} processed files, but {len(df_info)} synced and {len(df_fail)} failed")
    else:
        print(f"Success: {len(df_all)} processed files, {len(df_info)} synced, and {len(df_fail)} failed.")
    return rmse_list

def FramesShift9(csvlog, inpath, outpath, subjects, activity, activity_legend, outputfilename=None, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    csvlogfile = os.path.join(outpath, csvlog)
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath

    rmse_list = []
    for i, subject in enumerate(subjects):
        dfmot = None
        dfcsv = None

        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

            # Log if file not found
            if not os.path.exists(inpathmotfull):
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "File not found")
                continue  # Continue to the next trial

            folder = os.path.join(inpath, subject)

            # Load data
            dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
            allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
            allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')

        # Extract joint data
        jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
        jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
        jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)

        # 2) Downsample IMU to 30 fps and interpolate video (remove zeros)
        try:
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during interpolation or downsampling: {str(e)}")
            continue

        # 3) Smooth both signals
        try:
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal smoothing: {str(e)}")
            continue

        # 4) Compute RMSE of signals
        try:
            SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
            rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during RMSE calculation: {str(e)}")
            continue

        # 5) Center signals in mean for better synchronization
        try:
            jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
            jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
            rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal centering: {str(e)}")
            continue 

        # 6) Shift and cut signals to find ideal synchronization
        try:
            rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
            remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                jointangle_imus_centered, jointangle_video_centered, maxshift=MAX_SYNC_OVERLAP, fitlength=RMSE_SAMPLES)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during synchronization: {str(e)}")
            continue

        # Log and cut frames if necessary
        if bool_cut_imu:
            rm_samples_imu_minrmse = remove_samples_imu
            rm_samples_video_minrmse = 0
        elif bool_cut_video:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = remove_samples_video
        else:
            rm_samples_imu_minrmse = 0
            rm_samples_video_minrmse = 0

        if FINAL_LENGTH is None:
            FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse, len(jointangle_video_centered) - rm_samples_video_minrmse)

        try:
            jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                jointangle_imus_centered, jointangle_video_centered, rm_samples_imu_minrmse, rm_samples_video_minrmse, max_length=FINAL_LENGTH)
            if rmse_shift == 0:
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'rmse=0')
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during signal shifting: {str(e)}")
            continue

        # Log synchronization adjustments
        if remove_samples_imu > 0:
            imufilenameRAW = os.path.join(folder, subject + '_' + activity + '_' + trial + '.raw')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameRAW, 'raw', remove_samples_imu, rmse_centered, rmse_shift)

            imufilenameMOT = os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, imufilenameMOT, 'mot', remove_samples_imu, rmse_centered, rmse_shift)

        if remove_samples_video > 0:
            videofilenameMP4 = os.path.join(folder, subject + '_' + activity + '_' + trial + '.mp4')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameMP4, 'mp4', remove_samples_video, rmse_centered, rmse_shift)

            videofilenameCSV = os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv')
            addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, videofilenameCSV, 'csv', remove_samples_video, rmse_centered, rmse_shift)

        rmse_list.append(rmse_shift)

    # === Add Validation for File Length Consistency ===
    df_info = pd.read_csv(os.path.join(outpath, 'infoToSync.csv')).sort_values(by='Subject')
    df_info.to_csv(os.path.join(outpath, 'infoToSync.csv'))
    try:
        df_fail = pd.read_csv(os.path.join(outpath, 'failedSync.csv')).sort_values(by='Subject')
        df_fail.to_csv(os.path.join(outpath, 'failedSync.csv'))
    except:
        df_fail = pd.DataFrame()
    df_all = pd.read_csv(os.path.join(outpath, 'allfiles.csv')).sort_values(by='Subject')
    df_all.to_csv(os.path.join(outpath, 'allfiles.csv'))

    total_synced_failed = len(df_info) + len(df_fail)
    if len(df_all) != total_synced_failed:
        raise ValueError(f"File count mismatch: {len(df_all)} processed files, but {len(df_info)} synced and {len(df_fail)} failed")
    else:
        print(f"Success: {len(df_all)} processed files, {len(df_info)} synced, and {len(df_fail)} failed.")
    return rmse_list



def FramesShift10(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath
    csvlogfile = os.path.join(outpath,csvlog)

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # 1) Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

                # Skip logging missing files to failedSync.csv
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                print(f"Skipping missing files for Trial {trial}, Subject {subject}.")
                continue  # Skip to the next trial without logging to failedSync.csv
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')

                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue
        try: 
            jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
            jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
            jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
        
            # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
            # 3) Smooth both signals
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
            
            # 4) Compute RMSE of signals
            SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
            rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
        
            # 5) Center signals in mean for better synchronization
            jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
            jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
            rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
        
            # 6) Shift and cut signals to find ideal synchronization 
            rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
            remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)
            if bool_cut_imu:
                rm_samples_imu_minrmse=remove_samples_imu
                rm_samples_video_minrmse=0    
            elif bool_cut_video:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=remove_samples_video
            else:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=0

            if FINAL_LENGTH is None:
                print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
                print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
                FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
                print(FINAL_LENGTH)
            jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)

            #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

            # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
            if remove_samples_imu>0:
                imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
                
                imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)
            
            if remove_samples_video>0:
                videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)

                videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

            rmse_list.append(rmse_shift)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during processing: {str(e)}")
        continue

    return rmse_list

def FramesShift11(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath
    rmse_df = pd.DataFrame(columns=['Subject', 'Activity', 'Trial', 'File', 'Type', 'RMSE shift'])
    csvlogfile = os.path.join(outpath,csvlog)

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

                # Skip logging missing files to failedSync.csv
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                print(f"Skipping missing files for Trial {trial}, Subject {subject}.")
                continue  # Skip to the next trial without logging to failedSync.csv
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue


        try: 
            # 1) Get joint angle from IMU and video data
            jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
            jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
            jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 1")

        try:    
            # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 2")

        try:
            # 3) Smooth both signals
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 3")
            
        try:
            # 4) Compute RMSE of signals
            SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
            rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 4")

        try:
            # 5) Center signals in mean for better synchronization
            jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
            jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
            rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 5")

        try:
            # 6) Shift and cut signals to find ideal synchronization 
            rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
            remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)

            if bool_cut_imu == False and bool_cut_video == False:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Minimum RMSE not found")
            
            if bool_cut_imu:
                rm_samples_imu_minrmse=remove_samples_imu
                rm_samples_video_minrmse=0    
            elif bool_cut_video:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=remove_samples_video
            else:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=0

            if FINAL_LENGTH is None:
                print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
                print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
                FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
                print(FINAL_LENGTH)
            jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)

            #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

            # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
            if remove_samples_imu>0:
                imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
                
                imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)
                rmse_df =['Subject', 'Activity', 'Trial', 'File', 'Type', 'RMSE shift']
            if remove_samples_video>0:
                videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)

                videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

            rmse_list.append(rmse_shift)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 6")
        # except Exception as e:
        #     log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during processing: {str(e)}")
        continue
    return rmse_list


def FramesShift12(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath
    rmse_df = pd.DataFrame(columns=['Subject', 'Activity', 'Trial', 'File', 'Type', 'RMSE shift', 'Remove Samples'])
    csvlogfile = os.path.join(outpath,csvlog)
    rmse_path = os.path.join(outpath, 'rmse_df.csv')

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

                # Skip logging missing files to failedSync.csv
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                print(f"Skipping missing files for Trial {trial}, Subject {subject}.")
                continue  # Skip to the next trial without logging to failedSync.csv
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue


        try: 
            # 1) Get joint angle from IMU and video data
            jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
            jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
            jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 1")

        try:    
            # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 2")

        try:
            # 3) Smooth both signals
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 3")
            
        try:
            # 4) Compute RMSE of signals
            SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
            rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 4")

        try:
            # 5) Center signals in mean for better synchronization
            jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
            jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
            rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 5")

        try:
            # 6) Shift and cut signals to find ideal synchronization 
            rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
            remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)

            if bool_cut_imu == False and bool_cut_video == False:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Minimum RMSE not found")
            
            if bool_cut_imu:
                rm_samples_imu_minrmse=remove_samples_imu
                rm_samples_video_minrmse=0    
            elif bool_cut_video:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=remove_samples_video
            else:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=0

            if FINAL_LENGTH is None:
                print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
                print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
                FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
                print(FINAL_LENGTH)
            jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)

            #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

            # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
            if remove_samples_imu>0:

                imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
                imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)

                dictChanges = {
                    'Subject':[subject],
                    'Activity':[activity],
                    'Trial':[trial],
                    'File':[imufilenameMOT],
                    'Type':['mot'],
                    'RMSE shift': [rmse_shift],
                    'Remove Samples':[remove_samples_imu]
                    }
                entry=pd.DataFrame.from_dict(dictChanges)
                rmse_df = pd.concat([rmse_df,entry],ignore_index=True)
                rmse_df.to_csv(rmse_path,index=False,mode='w+')

            if remove_samples_video>0:

                videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)
                videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
                addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

                dictChanges = {
                    'Subject':[subject],
                    'Activity':[activity],
                    'Trial':[trial],
                    'File':[videofilenameCSV],
                    'Type':['csv'],
                    'RMSE shift': [rmse_shift],
                    'Remove Samples': [remove_samples_video]
                    }
                entry=pd.DataFrame.from_dict(dictChanges)
                rmse_df = pd.concat([rmse_df,entry],ignore_index=True)
                rmse_df.to_csv(rmse_path,index=False,mode='w+')


            rmse_list.append(rmse_shift)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 6")
        # except Exception as e:
        #     log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during processing: {str(e)}")
        continue
    return rmse_list


def FramesShift13(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath
    rmse_df = pd.DataFrame(columns=['Subject', 'Activity', 'Trial', 'File', 'Type', 'RMSE shift', 'Remove Samples'])
    csvlogfile = os.path.join(outpath,csvlog)
    rmse_path = os.path.join(outpath, 'rmse_df.csv')

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

                # Skip logging missing files to failedSync.csv
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                print(f"Skipping missing files for Trial {trial}, Subject {subject}.")
                continue  # Skip to the next trial without logging to failedSync.csv
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
                allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            print("Not found")
            continue


        try: 
            # 1) Get joint angle from IMU and video data
            jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
            jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
            jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 1")

        try:    
            # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 2")

        try:
            # 3) Smooth both signals
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 3")
            
        try:
            # 4) Compute RMSE of signals
            SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
            rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 4")

        try:
            # 5) Center signals in mean for better synchronization
            jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
            jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
            rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 5")

        try:
            # 6) Shift and cut signals to find ideal synchronization 
            rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
            remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)

            if bool_cut_imu == False and bool_cut_video == False:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Minimum RMSE not found")
            
            if bool_cut_imu:
                rm_samples_imu_minrmse=remove_samples_imu
                rm_samples_video_minrmse=0    
            elif bool_cut_video:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=remove_samples_video
            else:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=0

            if FINAL_LENGTH is None:
                print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
                print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
                FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
                print(FINAL_LENGTH)
            jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)

            #rmse_shift_real=signalutil.calcRMSE(jointangle_imus_shift[:RMSE_SAMPLES],jointangle_video_shift[:RMSE_SAMPLES])

            # # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
            # if remove_samples_imu>0:

            imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
            imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)

            dictChanges = {
                'Subject':[subject],
                'Activity':[activity],
                'Trial':[trial],
                'File':[imufilenameMOT],
                'Type':['mot'],
                'RMSE shift': [rmse_shift],
                'Remove Samples':[remove_samples_imu]
                }
            entry=pd.DataFrame.from_dict(dictChanges)
            rmse_df = pd.concat([rmse_df,entry],ignore_index=True)
            rmse_df.to_csv(rmse_path,index=False,mode='w+')

            # if remove_samples_video>0:

            videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)
            videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)

            dictChanges = {
                'Subject':[subject],
                'Activity':[activity],
                'Trial':[trial],
                'File':[videofilenameCSV],
                'Type':['csv'],
                'RMSE shift': [rmse_shift],
                'Remove Samples': [remove_samples_video]
                }
            entry=pd.DataFrame.from_dict(dictChanges)
            rmse_df = pd.concat([rmse_df,entry],ignore_index=True)
            rmse_df.to_csv(rmse_path,index=False,mode='w+')


            rmse_list.append(rmse_shift)

            
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 6")
        # except Exception as e:
        #     log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error during processing: {str(e)}")
        continue
    return rmse_list

def FramesShift14(csvlog,inpath,outpath,subjects,activity,activity_legend,outputfilename=None,RMSE_SAMPLES=200,MAX_SYNC_OVERLAP=15,FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync.csv')  # Log for failed synchronizations
    # allfiles_csv_logfile = os.path.join(outpath, 'allfiles.csv')  # Log for all files in inpath
    # rmse_df = pd.DataFrame(columns=['Subject', 'Activity', 'Trial', 'File', 'Type', 'RMSE shift', 'Remove Samples'])
    csvlogfile = os.path.join(outpath,csvlog)
    # rmse_path = os.path.join(outpath, 'rmse_df.csv')

    rmse_list =[]
    for i,subject in enumerate(subjects):

        dfmot = None
        dfcsv = None

        # Load/compute imu and video joint's angle signals
        #Try to open the first existing trial for that subject

        for trial in ["T01","T02","T03","T04","T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

                # Skip logging missing files to failedSync.csv
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                # print(f"Skipping missing files for Trial {trial}, Subject {subject}.")
                continue  
            else:
                folder = os.path.join(inpath,subject)
                # print("READING: "+inpathmotfull[:-4])
                dfmot,dfcsv = fileutil.readMOTandCSV(folder,subject,activity,trial)
                # allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathmotfull, 'mot')
                # allfiles_sync(allfiles_csv_logfile, subject, activity, trial, inpathcsvfull, 'csv')
                break #limit to the first existing trial
        if dfmot is None or dfcsv is None:
            # print("Not found")
            continue


        try: 
            # 1) Get joint angle from IMU and video data
            jointMot,bonesCSV=getMainJointFromMotAndMainBonesFromCSV(dfmot,dfcsv,activity)
            jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot,jointMot)
            jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 1")

        try:    
            # 2) Downsample imu to 30 fps and interpolate video (remove zeros)
            jointangle_video_inter = signalutil.fill_nan(jointangle_video)
            jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus,50,30)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 2")

        try:
            # 3) Smooth both signals
            jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
            jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 3")
            
        try:
            # 4) Compute RMSE of signals
            SUBARRAY_SAMPLES = RMSE_SAMPLES+MAX_SYNC_OVERLAP
            rmse_smooth=signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES],jointangle_video_cutfilt[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 4")

        try:
            # 5) Center signals in mean for better synchronization
            jointangle_imus_centered=signalutil.centerSignalInMean(jointangle_imus_cutfilt,samples=SUBARRAY_SAMPLES)
            jointangle_video_centered=signalutil.centerSignalInMean(jointangle_video_cutfilt,samples=SUBARRAY_SAMPLES)
            rmse_centered=signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES],jointangle_video_centered[:RMSE_SAMPLES])
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 5")

        try:
            # 6) Shift and cut signals to find ideal synchronization 
            rmse_original,remove_samples_imu,minimum_rmse_imu,bool_cut_imu,\
            remove_samples_video,minimum_rmse_video,bool_cut_video=getSamplesToSynchronize(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                maxshift=MAX_SYNC_OVERLAP,
                                                                fitlength=RMSE_SAMPLES)

            if bool_cut_imu == False and bool_cut_video == False:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Minimum RMSE not found")
            
            if bool_cut_imu:
                rm_samples_imu_minrmse=remove_samples_imu
                rm_samples_video_minrmse=0    
            elif bool_cut_video:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=remove_samples_video
            else:
                rm_samples_imu_minrmse=0
                rm_samples_video_minrmse=0

            if FINAL_LENGTH is None:
                print("IMUS: ",len(jointangle_imus_centered),rm_samples_imu_minrmse)
                print("VIDEO: ",len(jointangle_video_centered),rm_samples_video_minrmse)
                FINAL_LENGTH=min(len(jointangle_imus_centered)-rm_samples_imu_minrmse,len(jointangle_video_centered)-rm_samples_video_minrmse)
                print(FINAL_LENGTH)
            jointangle_imus_shift, jointangle_video_shift,rmse_shift = SynchronizeAndCutSignals(
                                                                jointangle_imus_centered,
                                                                jointangle_video_centered,
                                                                rm_samples_imu_minrmse,
                                                                rm_samples_video_minrmse,
                                                                max_length=FINAL_LENGTH)

            # # LOG SYNCHRONIZATION ADJUSTMENTS TO MODIFY FILES
            # if remove_samples_imu>0:

            imufilenameRAW = os.path.join(folder,subject+'_'+activity+'_'+trial+'.raw')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameRAW,'raw',remove_samples_imu,rmse_centered,rmse_shift)
            imufilenameMOT = os.path.join(folder,'ik_'+subject+'_'+activity+'_'+trial+'.mot')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,imufilenameMOT,'mot',remove_samples_imu,rmse_centered,rmse_shift)

            # if remove_samples_video>0:

            videofilenameMP4 = os.path.join(folder,subject+'_'+activity+'_'+trial+'.mp4')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameMP4,'mp4',remove_samples_video,rmse_centered,rmse_shift)
            videofilenameCSV = os.path.join(folder,subject+'_'+activity+'_'+trial+'.csv')
            addFramesShiftToCSVLog(csvlogfile,subject,activity,trial,videofilenameCSV,'csv',remove_samples_video,rmse_centered,rmse_shift)


            rmse_list.append(rmse_shift)

            
        except Exception as e:
            log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, f"Error at step 6")

        continue
    return rmse_list

def FramesShift15(csvlog, inpath, outpath, subjects, activity, activity_legend, outputfilename=None, RMSE_SAMPLES=200, MAX_SYNC_OVERLAP=15, FINAL_LENGTH=None):
    failed_csv_logfile = os.path.join(outpath, 'failedSync_all.csv')  # Log for failed synchronizations
    csvlogfile = os.path.join(outpath, csvlog)

    rmse_list = []
    
    for i, subject in enumerate(subjects):

        # Iterate over all trials for each subject
        for trial in ["T01", "T02", "T03", "T04", "T05"]:
            motsubjacttrial = subject + "_" + activity + "_" + trial
            motfilename = 'ik_' + motsubjacttrial + ".mot"
            csvfilename = motsubjacttrial + ".csv"
            inpathmotfull = os.path.join(inpath, subject, motfilename)
            inpathcsvfull = os.path.join(inpath, subject, csvfilename)

            # Check if both files exist
            if not os.path.exists(inpathmotfull) or not os.path.exists(inpathcsvfull):
                continue  

            # Load data
            folder = os.path.join(inpath, subject)
            try:
                dfmot, dfcsv = fileutil.readMOTandCSV(folder, subject, activity, trial)
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Failed to read data")
                continue
            
            try:
                # 1) Get joint angle from IMU and video data
                jointMot, bonesCSV = getMainJointFromMotAndMainBonesFromCSV(dfmot, dfcsv, activity)
                jointangle_imus = fileutil.getJointAngleMotAsNP(dfmot, jointMot)
                jointangle_video = fileutil.getJointAngleCsvAsNP(bonesCSV)
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 1")
                continue

            try:    
                # 2) Downsample imu to 30 fps and interpolate video
                jointangle_video_inter = signalutil.fill_nan(jointangle_video)
                jointangle_imus_cutdown = signalutil.downsampleSignal(jointangle_imus, 50, 30)
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 2")
                continue

            try:
                # 3) Smooth both signals
                jointangle_imus_cutfilt = signalutil.applyMovingAverageFilter(jointangle_imus_cutdown)
                jointangle_video_cutfilt = signalutil.applyMovingAverageFilter(jointangle_video_inter)
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 3")
                continue

            try:
                # 4) Compute RMSE of signals
                SUBARRAY_SAMPLES = RMSE_SAMPLES + MAX_SYNC_OVERLAP
                rmse_smooth = signalutil.calcRMSE(jointangle_imus_cutfilt[:RMSE_SAMPLES], jointangle_video_cutfilt[:RMSE_SAMPLES])
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 4")
                continue

            try:
                # 5) Center signals for better synchronization
                jointangle_imus_centered = signalutil.centerSignalInMean(jointangle_imus_cutfilt, samples=SUBARRAY_SAMPLES)
                jointangle_video_centered = signalutil.centerSignalInMean(jointangle_video_cutfilt, samples=SUBARRAY_SAMPLES)
                rmse_centered = signalutil.calcRMSE(jointangle_imus_centered[:RMSE_SAMPLES], jointangle_video_centered[:RMSE_SAMPLES])
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 5")
                continue

            try:
                # 6) Shift and cut signals for synchronization
                rmse_original, remove_samples_imu, minimum_rmse_imu, bool_cut_imu, \
                remove_samples_video, minimum_rmse_video, bool_cut_video = getSamplesToSynchronize(
                    jointangle_imus_centered,
                    jointangle_video_centered,
                    maxshift=MAX_SYNC_OVERLAP,
                    fitlength=RMSE_SAMPLES
                )

                if not bool_cut_imu and not bool_cut_video:
                    log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Minimum RMSE not found")
                    continue

                if bool_cut_imu:
                    rm_samples_imu_minrmse = remove_samples_imu
                    rm_samples_video_minrmse = 0    
                elif bool_cut_video:
                    rm_samples_imu_minrmse = 0
                    rm_samples_video_minrmse = remove_samples_video
                else:
                    rm_samples_imu_minrmse = 0
                    rm_samples_video_minrmse = 0

                if FINAL_LENGTH is None:
                    FINAL_LENGTH = min(len(jointangle_imus_centered) - rm_samples_imu_minrmse,
                                       len(jointangle_video_centered) - rm_samples_video_minrmse)

                jointangle_imus_shift, jointangle_video_shift, rmse_shift = SynchronizeAndCutSignals(
                    jointangle_imus_centered,
                    jointangle_video_centered,
                    rm_samples_imu_minrmse,
                    rm_samples_video_minrmse,
                    max_length=FINAL_LENGTH
                )

                # Log synchronization adjustments
                addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, os.path.join(folder, 'ik_' + subject + '_' + activity + '_' + trial + '.mot'), 'mot', remove_samples_imu, rmse_centered, rmse_shift)
                addFramesShiftToCSVLog(csvlogfile, subject, activity, trial, os.path.join(folder, subject + '_' + activity + '_' + trial + '.csv'), 'csv', remove_samples_video, rmse_centered, rmse_shift)

                rmse_list.append(rmse_shift)
            
            except Exception as e:
                log_failed_sync(failed_csv_logfile, subject, activity, trial, inpathmotfull, "Error at step 6")
                continue

    return rmse_list
