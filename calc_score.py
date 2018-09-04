import json
import os
import os.path as op
import numpy as np
import json
import cv2

connections=[[0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],
          [0,17],[17,18],[18,19],[19,20]]
colors = [[255, 0, 0], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], 
              [0, 255, 0],   [0, 255, 85], 
              [0, 255, 170], [0, 255, 255], 
              [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255],
              [255, 0, 0], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], 
              [0, 255, 0],   [0, 255, 85],
              [255, 0, 0], [255, 85, 0]]
              
def txt2json(label_dir, json_name):
    label_files=os.listdir(label_dir)
    dataset=[]
    for file in label_files:
        entry={}
        name=file[:file.rfind('.')]
        entry['img_name']=name+'.jpg'
        with open(op.join(label_dir,file), 'r') as f:
            lines=f.readlines()
            xys=[list(map(float, line.split(' '))) for line in lines]
#            xys=np.asarray(xys).reshape(-1,3)
            entry['joints']=xys
            dataset.append(entry)
    with open(json_name, 'w') as f:
        f.write(json.dumps(dataset))

def vis(json_name):
    img_dir='images'
    vis_dir='vis'
    dataset=json.load(open(json_name, 'r'))
    for entry in dataset:
        img_name=entry['img_name']
        image=cv2.imread(op.join(img_dir,img_name))
        joints=entry['joints']
        for conn in connections:
            start=conn[0]
            end=conn[1]
            jo_start=np.asarray(joints[start]).astype(np.int32)
            jo_end=np.asarray(joints[end]).astype(np.int32)
            if jo_start[2]==1:
                cv2.circle(image, (jo_start[0],jo_start[1]), 3, colors[start], -1)
            if jo_end[2]==1:
                cv2.circle(image, (jo_end[0],jo_end[1]), 3, colors[end], -1)
            if jo_start[2]==1 and jo_end[2]==1:
                cv2.line(image, (jo_start[0],jo_start[1]),(jo_end[0],jo_end[1]),colors[end],2)
            cv2.imwrite(op.join(vis_dir, img_name),image)
            
def calc_scale(entry):
    joints=np.asarray(entry['joints'])
    joints=joints[joints[:,2]>0]
    min_x=min(joints[:,0])
    min_y=min(joints[:,1])
    max_x=max(joints[:,0])
    max_y=max(joints[:,1])
    return (max_x-min_x)*(max_y-min_y)
    
def calc_oks(pred_entry, ref_entry):
    joints_pred=np.asarray(pred_entry['joints'])
    joints_ref=np.asarray(ref_entry['joints'])
    valid_inds=joints_ref[:,2]>0
    joints_pred_valid=joints_pred[valid_inds]
    joints_ref_valid=joints_ref[valid_inds]
    dis = np.sum((joints_ref_valid[:,:2]- joints_pred_valid[:,:2])**2, axis=1)
#    print(dis)
    scale=calc_scale(ref_entry)
    delta=0.002
    oks = np.mean(np.exp(-dis/(2*delta**2*(scale+1)**2)))
#    oks = np.mean(np.exp(-dis))
    return oks
    
def calc_ap(pred_json, ref_json):
    ref_data=json.load(open(ref_json, 'r'))
    pred_data=json.load(open(pred_json, 'r'))
    sum_oks=0
    OKS=0.5
    for i in range(len(ref_data)):
        ref_entry=ref_data[i]
        img_name=ref_entry['img_name']
        for j in range(len(pred_data)):
            pred_entry=pred_data[j]
            if pred_entry['img_name']==img_name:
                oks=calc_oks(pred_entry, ref_entry)
                if oks>=OKS:
                    sum_oks+=oks
#                print('sample calculated')
                break
    sum_oks/=len(ref_data)
    print('mAP: {}'.format(sum_oks))
    
if __name__=='__main__':
#    txt2json('labels', 'gt.json')
    pred_json='./pred_handbody.json'
    ref_json='./gt.json'
    calc_ap(pred_json, ref_json)
#    vis(ref_json)