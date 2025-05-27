import json
from statistics import mean
import numpy as np 
import math
import numpy as np
from sklearn.metrics import roc_auc_score


# good model: 3 35

file_list = []

a =[1,3,4,11,27,30,32,45,46,50] # 暂时的最终效果
# a =[1,3,4,27,30,32,45,46, 50] 

a = list(range(1, 51))


a = [7,8,19,21,22,24,29,43,50] # zd

for i in a:
    file_list.append('bysy_infer_json_2task_zd_r{}.jsonl'.format(i))
    # file_list.append('bysy_infer_json_2task_zd_missimg1_r{}.jsonl'.format(i))
    # file_list.append('bysy_infer_json_2task_zd_missimg2_r{}.jsonl'.format(i))
    # file_list.append('bysy_infer_json_2task_zd_missimg12_r{}.jsonl'.format(i))
    # file_list.append('bysy_infer_json_2task_zd_misstabular_r{}.jsonl'.format(i))

true_positive = [] # 把慢性判别为慢性TP
false_positive = [] # 把无排异判别为慢性FP
true_negative = []  # 把无排异判别无排异TN
false_negative = [] # 把慢性判别无排异FN

result_all = [] # 10次采样的准确率
sensitivity_all = [] # 10次采样的敏感度
specificity_all = [] # 10次采样的特异度
f1_score_all = [] # 10次采样的F1
auc_all = [] # 10次采样的F1
# 91：9的数据配比

file_new = []


# file_list=['/public/mmllm/caolili/Qwen-VL-old-bysy/eval_med/bysy_infer_json_20240710_latest6_r4.jsonl']

# file_list = [ 'bysy_infer_json_20240711_latest7_r8.jsonl', 'bysy_infer_json_20240711_latest7_r19.jsonl', 'bysy_infer_json_20240711_latest7_r23.jsonl', 'bysy_infer_json_20240711_latest7_r26.jsonl', 'bysy_infer_json_20240711_latest7_r28.jsonl', 'bysy_infer_json_20240711_latest7_r29.jsonl', 'bysy_infer_json_20240711_latest7_r30.jsonl', 'bysy_infer_json_20240711_latest7_r31.jsonl', 'bysy_infer_json_20240711_latest7_r40.jsonl', 'bysy_infer_json_20240711_latest7_r46.jsonl']  # 这个就是诊断模型在本数据集上的最终版本: 8\19\23\26\28\29\30\31\40\46
c=12
# file_list = [f'bysy_infer_json_2task_zhenduan_r{c}.jsonl']
# file_list=['bysy_infer_json_20240731_onemodaltest_r8.jsonl'] # r8

y_true = []
y_pred = []
for file in file_list:
    print(111111111111111,file)

    correct = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    with open(file, 'r') as f:
        data = json.load(f)
        for i in data:
            t1 = i['answer'].split('：')[-1].strip()
            t2 = i['completion'].split('：')[-1].strip()
            # print(i['completion'], i['answer'], 1111111, t1, t2)
            if t2=='未发生排斥反应':
                t2 = '未发生排异'
            elif t2=='未发生慢性移植物抗宿主病':
                t2 = '未发生排异'
            elif t2=='检查结果为慢性移植物抗宿主病':
                t2 = '慢性移植物抗宿主病'
            elif t2=='检查结果为未发生慢性移植物抗宿主病':
                t2 = '未发生排异'
            
            if  t2 not in ['慢性移植物抗宿主病', '未发生排异']:
                print('buggggggggggg', t2)
                continue
            
            if t1==t2:
                correct += 1
            else:
                print(555555555, i['question_id'], t1, t2)
                
            if t1=='慢性移植物抗宿主病':
                y_true.append(1)
            elif t1=='未发生排异':
                y_true.append(0)
            if t2=='慢性移植物抗宿主病':
                y_pred.append(1) # 只要输出，概率都是1
            elif t2=='未发生排异':
                y_pred.append(0)
                
            if t1=='慢性移植物抗宿主病' and t2=='慢性移植物抗宿主病':
                TP += 1
            elif t1=='未发生排异' and t2=='慢性移植物抗宿主病':
                FP += 1
            elif t1=='未发生排异' and t2=='未发生排异':
                TN += 1
            elif t1=='慢性移植物抗宿主病' and t2=='未发生排异':
                FN += 1
    print(TP+FP+TN+FN)
    # assert TP+FP+TN+FN==277
    true_positive.append(TP)
    false_positive.append(FP)
    true_negative.append(TN)
    false_negative.append(FN)
    
    sensitivity = TP/(TP+FN)
    sensitivity_all.append(sensitivity)
    specificity = TN/(TN+FP)
    specificity_all.append(specificity)
    # precision = TP/(TP+FP)
    # recall = TP/(TP+FN)
    # f1_ = 2*precision*recall/(precision+recall)
    
    f1 = 2*TP/(2*TP+FP+FN)
    f1_score_all.append(f1)
    
    if (correct/len(data))>=0.9 and specificity>0.9 :
        file_new.append(file)
    
    result_all.append(correct/len(data)) 
    print('当前准确率：', correct/len(data), (TP+TN)/(TP+FP+TN+FN))
    print('当前灵敏度， 特异度, F1', sensitivity, specificity, f1)
    
    # 计算auc
    
    y = np.array(y_true)  # 真实值
    y_pred1 = np.array(y_pred)  # 预测值

    # 预测值是概率
    auc_score1 = roc_auc_score(y, y_pred1)
    auc_all.append(auc_score1)
    print('当前auc:', auc_score1) 
    
    
    
    f.close()
print('平均准确率：',mean(result_all), np.mean(result_all), np.var(result_all),  np.std(result_all,ddof=1))

# print(true_positive)
# print(false_positive)
# print(true_negative)
# print(false_negative)

print('平均灵敏度， 特异度, F1score', mean(sensitivity_all), mean(specificity_all), mean(f1_score_all))


sample_num = len(file_list)

# 计算95%置信区间
print('---------灵敏度')
bound1 = mean(sensitivity_all)+1.96*(np.std(sensitivity_all,ddof=1)/math.sqrt(sample_num))
bound2 = mean(sensitivity_all)-1.96*(np.std(sensitivity_all,ddof=1)/math.sqrt(sample_num))
print("{:.2f}".format(mean(sensitivity_all)*100), "({:.2f}-".format(bound2*100),"{:.2f})".format(bound1*100),"{:.4f}".format(bound1-bound2))

# 计算95%置信区间
print('---------特异度')
bound1 = mean(specificity_all)+1.96*(np.std(specificity_all,ddof=1)/math.sqrt(sample_num))
bound2 = mean(specificity_all)-1.96*(np.std(specificity_all,ddof=1)/math.sqrt(sample_num))
print("{:.2f}".format(mean(specificity_all)*100), "({:.2f}-".format(bound2*100),"{:.2f})".format(bound1*100),"{:.4f}".format(bound1-bound2))

# 计算95%置信区间
print('---------准确率')
bound1 = mean(result_all)+1.96*(np.std(result_all,ddof=1)/math.sqrt(sample_num))
bound2 = mean(result_all)-1.96*(np.std(result_all,ddof=1)/math.sqrt(sample_num))
print("{:.2f}".format(mean(result_all)*100), "({:.2f}-".format(bound2*100),"{:.2f})".format(bound1*100),"{:.4f}".format(bound1-bound2))


# 计算95%置信区间
print('---------auc score')
bound1 = mean(auc_all)+1.96*(np.std(auc_all,ddof=1)/math.sqrt(sample_num))
bound2 = mean(auc_all)-1.96*(np.std(auc_all,ddof=1)/math.sqrt(sample_num))
print("{:.2f}".format(mean(auc_all)*100), "({:.2f}-".format(bound2*100),"{:.2f})".format(bound1*100),"{:.4f}".format(bound1-bound2))


# 计算95%置信区间
print('---------F1 score')
bound1 = mean(f1_score_all)+1.96*(np.std(f1_score_all,ddof=1)/math.sqrt(sample_num))
bound2 = mean(f1_score_all)-1.96*(np.std(f1_score_all,ddof=1)/math.sqrt(sample_num))
print("{:.2f}".format(mean(f1_score_all)*100), "({:.2f}-".format(bound2*100),"{:.2f})".format(bound1*100),"{:.4f}".format(bound1-bound2))



print(file_new, len(file_new))

print(len(file_list))


