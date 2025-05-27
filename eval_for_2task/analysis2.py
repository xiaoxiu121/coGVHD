import json, re

f = 'bysy_infer_alignment.jsonl'



with open(f, 'r') as f:
    data = json.load(f)
    
    correct_num = 0
    all_num = 0
    for data_i in data:
        answer = data_i['answer']
        predic = data_i['prediction']
        
        r1 = re.findall(r'眼表疾病指数量表(.*?),', answer)
        r1_p = re.findall(r'眼表疾病指数量表(.*?),', predic)
        
        
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
            
        # print(r1, r1_p, data_i['question_id'])
        
        r1 = re.findall(r'角膜荧光染色评分(.*?),', answer)
        r1_p = re.findall(r'角膜荧光染色评分(.*?),', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'泪膜破裂时间(.*?),', answer)
        r1_p = re.findall(r'泪膜破裂时间(.*?),', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'泪河高度(.*?),', answer)
        r1_p = re.findall(r'泪河高度(.*?),', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        
        r1 = re.findall(r'泪液分泌实验(.*?),', answer)
        r1_p = re.findall(r'泪液分泌实验(.*?),', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        
        r1 = re.findall(r'([\u4e00-\u9fa5]+)皮肤排异', answer)
        r1_p = re.findall(r'([\u4e00-\u9fa5]+)皮肤排异', predic)
        # print(r1, r1_p, data_i['question_id'])
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'([\u4e00-\u9fa5]+)口腔排异', answer)
        r1_p = re.findall(r'([\u4e00-\u9fa5]+)口腔排异', predic)
        # print(r1, r1_p, data_i['question_id'])
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'([\u4e00-\u9fa5]+)肠道排异', answer)
        r1_p = re.findall(r'([\u4e00-\u9fa5]+)肠道排异', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'([\u4e00-\u9fa5]+)肺排异', answer)
        r1_p = re.findall(r'([\u4e00-\u9fa5]+)肺排异', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        r1 = re.findall(r'([\u4e00-\u9fa5]+)肝排异', answer)
        r1_p = re.findall(r'([\u4e00-\u9fa5]+)肝排异', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        
        
        r1 = re.findall(r'哭时(.*?)使用电子产品类型', answer)
        r1_p = re.findall(r'哭时(.*?)使用电子产品类型', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        
        
        r1 = re.findall(r'使用电子产品类型为(.*?)每天平均电子产', answer)
        r1_p = re.findall(r'使用电子产品类型为(.*?)每天平均电子产', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
        

        
        r1 = re.findall(r'每天平均电子产品使用时间(.*?)。', answer)
        r1_p = re.findall(r'每天平均电子产品使用时间(.*?)。', predic)
        if r1==[]  : # 原始数据中不存在这个指标;预测也没有
            pass
        else:
            all_num += 1
            if r1==r1_p:
                correct_num+= 1     
                
                
print(correct_num, correct_num/all_num)
        
        
        
        
                