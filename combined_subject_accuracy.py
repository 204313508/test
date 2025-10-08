import os
import pandas as pd
import re
import argparse

def process_files(folder_path, prefix="mmlu"):
    # 匹配文件名的正则表达式
    # Build regex pattern dynamically based on the given prefix
    pattern = re.compile(rf"{re.escape(prefix)}_([a-z]+)_llm_result[^_]*_subject_accuracy(?:\d*)\.csv")
    
    # 存储所有找到的数据
    all_data = []
    
    # 遍历文件夹中的所有CSV文件
    for filename in os.listdir(folder_path):
        if match := pattern.match(filename):
            data_type = match.group(1)  # 提取类型（audio/text/image）
            file_path = os.path.join(folder_path, filename)
            
            try:
                df = pd.read_csv(file_path)
                # 添加类型和文件名作为新列
                df['type'] = data_type
                all_data.append(df)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    if not all_data:
        print("未找到符合条件的CSV文件")
        return
    
    # 合并所有数据
    full_df = pd.concat(all_data, ignore_index=True)
    
    # 按类型和学科分组计算平均准确率
    # Calculate mean and variance of accuracy for each type and subject
    agg_df = full_df.groupby(['type', 'subject'])['accuracy'].agg(['mean', 'var']).reset_index()
    agg_df.rename(columns={'mean': 'average_accuracy', 'var': 'variance'}, inplace=True)
    
    # 保存结果到新的CSV文件
    output_path = os.path.join(folder_path, "combined_subject_accuracy.csv")
    agg_df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine subject accuracy CSV files and compute statistics.")
    parser.add_argument("folder_path", help="Path to the folder containing CSV files.")
    parser.add_argument("--prefix", default="mmlu", help="Filename prefix to match. Default: mmlu")
    args = parser.parse_args()

    output = process_files(args.folder_path, args.prefix)
    if output:
        print(f"结果已保存到: {output}")
