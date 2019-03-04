import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio 

subject_list=['100307','102816','104012','105923','106521','108323','109123','111514','112920','113922','116726','125525','133019','140117','146129','149741',
'151526','156334','158136','162026','162935','164636','166438','169040','172029','175237','175540','177746','182840','185442','189349','191033','191437','191841',
'192641','195041','198653','200109','204521','205119','212318','212823','214524','223929','248339','250427','255639','257845','283543','293748','352738','353740',
'358144','406836','433839','500222','512835','555348','568963','581450','599671','601127','660951','662551','665254','667056','679770','680957','706040','707749',
'715950','725751','735148','783462','814649','825048','872764','877168','891667','898176','912447','917255','990366']
for subject in subject_list:
	x=sio.loadmat(os.path.join('/Volumes/My Passport/MEG/Matlab/'+subject+'_wrkmem_data_for_Mapper.mat'),struct_as_record=False,squeeze_me=True)
	data=x['concatdata_wrkmem']
	data=np.transpose(data)
	trial_info=x['trialInfo_table_wrkmem'][:,3:5]
	numbr_of_trials=trial_info.shape[0]
	numbr_of_channels=data.shape[1]
	size_of_trial=int(data.shape[0]/numbr_of_trials)
	trial_face0B=np.zeros((size_of_trial,numbr_of_channels))
	trial_face2B=np.zeros((size_of_trial,numbr_of_channels))
	trial_tool0B=np.zeros((size_of_trial,numbr_of_channels))
	trial_tool2B=np.zeros((size_of_trial,numbr_of_channels))
	f0B=0
	f2B=0
	t0B=0
	t2B=0
	for k,(i,j) in enumerate(trial_info):
		if (i,j)==(1,1):
			trial_face0B=trial_face0B+data[640*k:640*(k+1),:]
			f0B=f0B+1
		elif (i,j)==(1,2):
			trial_face2B=trial_face2B+data[640*k:640*(k+1),:]
			f2B=f2B+1
		elif (i,j)==(2,1):
			trial_tool0B=trial_tool0B+data[640*k:640*(k+1),:]
			t0B=t0B+1
		else:
			trial_tool2B=trial_tool2B+data[640*k:640*(k+1),:]
			t2B=t2B+1
	avg_face0B=(1/f0B)*trial_face0B
	avg_face2B=(1/f2B)*trial_face2B
	avg_tool0B=(1/t0B)*trial_tool0B
	avg_tool2B=(1/t2B)*trial_tool2B
	np.savetxt('/Volumes/My Passport/MEG/py_output/wrkmem_'+subject+'_avg_face0B.txt',avg_face0B,delimiter=',')
	np.savetxt('/Volumes/My Passport/MEG/py_output/wrkmem_'+subject+'_avg_face2B.txt',avg_face2B,delimiter=',')
	np.savetxt('/Volumes/My Passport/MEG/py_output/wrkmem_'+subject+'_avg_tool0B.txt',avg_tool0B,delimiter=',')
	np.savetxt('/Volumes/My Passport/MEG/py_output/wrkmem_'+subject+'_avg_tool2B.txt',avg_tool2B,delimiter=',')
