import json
import numpy as np
from sklearn.metrics import accuracy_score
import re
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score

# 计算一致性相关系数（CCC）
def concordance_correlation_coefficient(x, y):
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # 计算协方差
    covariance = np.cov(x, y)[0][1]
    
    # 计算方差
    var_x = np.var(x)
    var_y = np.var(y)
    
    # 计算CCC
    ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)
    
    return ccc
    
# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


dataset = 'CMDC'

if dataset == 'CMDC':
    # 提取评估结果中的'评估结果'部分
    def extract_assessment_result(key, result):
        # 匹配 "评估结果:数字" 的格式
        match = re.search(r"评估结果[:：](-?\d+\.\d+)", result)
        if match:
            return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果没有匹配到，尝试其他可能格式
        match = re.search(r"评估结果为[:：]?(\d)", result)
        if match:
            return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果没有匹配到，尝试其他可能格式
        match = re.search(r"评估结果是[:：]?(\d)", result)  # 处理"估结果"的特殊情况
        if match:
             return round(float(match.group(1)))  # 四舍五入返回整数
    
        match = re.search(r"[为:：]?(\d)", result)  # 处理"估结果"的特殊情况
        if match:
             return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 如果都没有匹配到，打印结果供调试
        print(f"未匹配到评估结果：{result}")
        return 1.5  # 返回 None 表示未找到结果

else:
    def extract_assessment_result(key, result):
        # 1. 匹配标准格式 "Evaluation Result: 数字"（支持大小写）
        match = re.search(r"(?i)Evaluation (?i)Result: (-?\d+\.\d+)", result)  # (?i)表示忽略大小写
        if match:
            return round(float(match.group(1)))  # 四舍五入返回整数
    
        # 2. 匹配简化格式 "Result: 数字"（支持大小写）
        match = re.search(r"(?i)Result: (\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 3. 匹配可能的 "Score: 数字"（支持大小写）
        match = re.search(r"(?i)Score: (\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 4. 匹配孤立数字（兜底方案）
        match = re.search(r"(\d+\.?\d*)", result)
        if match:
            return round(float(match.group(1)))
    
        # 5. 如果都没有匹配到，打印结果供调试
        print(f"{key}未匹配到评估结果：{result}")
        return 1.5  # 返回均值补齐
        

# 主函数：处理JSON数据，计算预测结果和准确率
def process_data(file_path):
    data = read_json(file_path)  # 假设该函数已经定义并读取JSON数据

    # 用于存储真实标签
    user_labels = defaultdict(list)
    # 用于存储每个用户的预测结果
    user_scores = defaultdict(list)

    # 计算每个数据的结果
    for item in data:
        key = item['key']
        result = item['result']
        label = item['label']

        # 提取评估结果
        evaluation_result = extract_assessment_result(key, result)
            
        if evaluation_result is None:
            print('没有评估结果', key)
            continue  # 如果评估结果为空，跳过

        # 将评估结果添加到相应用户的评分列表中
        user_id = key.split('-')[0]  # 提取用户ID（如 HC01）
        # 添加预测结果
        user_scores[user_id].append(evaluation_result)
        # 添加真实label
        user_labels[user_id].append(label)

    print("预测结果（每个用户）：", user_scores)
    print("真实标签（每个用户）：", user_labels)

    # 计算用户总分
    total_user_scores = {user_id: sum(scores) for user_id, scores in user_scores.items()}
    total_user_labels = {user_id: sum(labels) for user_id, labels in user_labels.items()}

    # 提取预测值和真实值
    predicted_totals = np.array(list(total_user_scores.values()))
    true_totals = np.array(list(total_user_labels.values()))

    # 计算 RMSE 和 MAE
    rmse = np.sqrt(mean_squared_error(true_totals, predicted_totals))
    mae = mean_absolute_error(true_totals, predicted_totals)

    print(f"用户总分的 RMSE: {rmse:.4f}")
    print(f"用户总分的 MAE: {mae:.4f}")


    # 计算一致性相关系数（CCC）
    ccc_result = concordance_correlation_coefficient(true_totals, predicted_totals)
    print(f"一致性相关系数 CCC: {ccc_result:.4f}")

    # 二元分类
    binary_predictions = [1 if score >= 9 else 0 for score in predicted_totals]
    binary_labels = [1 if label >= 9 else 0 for label in true_totals]

    # print(total_user_scores,total_user_labels)

    precision = precision_score(binary_labels, binary_predictions, pos_label=1)
    recall = recall_score(binary_labels, binary_predictions, pos_label=1)
    f1 = f1_score(binary_labels, binary_predictions, pos_label=1)

    print(f"二元分类的 Precision: {precision:.4f}")#正样本预测正确的概率
    print(f"二元分类的 Recall: {recall:.4f}")
    print(f"二元分类的 F1 Score: {f1:.4f}")


# 示例调用
file_path = './test_results_with_generated_output.json'  # JSON文件路径

process_data(file_path)
