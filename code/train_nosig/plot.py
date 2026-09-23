## 所有绘图函数集中存放在这里
## Xie Yuxuan 2024-08-22 UTF-8
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



## 绘制训练和验证损失
## 两个曲线绘制在一张图上
def plot_train_and_val_loss(train_losses,val_losses,save_path):

    max_loss = max(train_losses.max(), val_losses.max())
    min_loss = min(train_losses.min(), val_losses.min())

    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='train loss', color='blue')
    plt.plot(val_losses, label='val loss', color='red')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(min_loss, max_loss * 1.1)
    plt.title('Train_Val_Loss')
    plt.savefig(os.path.join(save_path, 'train_val_loss.png'))



## 绘制学习率,暂时没有使用这个函数
def plot_lr(learning_rates,save_path):

    plt.figure(figsize=(12, 5))
    plt.plot(learning_rates, label='learning rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.savefig(os.path.join(save_path, 'learning_rate.png'))



## 绘制测试性能的相关图片
## 绘制在一个目录下保存的所有方法的结果文件，方法名就是文件夹名字
## 会绘制出不同signum下，RMSE随SNR变化（所有snapnum下），RMSE随snapnum变化（所有SNR下）
def plot_results_from_folders(root_path):
    
    method_data = {}  # 初始化一个空字典来存储各个方法的数据
    save_path = os.path.join(root_path, 'Fig')  # 获取绝对路径

    # 遍历每个方法文件夹,获取其处理结果
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):  # 确保是目录
            csv_file = os.path.join(folder_path, 'results.csv')
            if os.path.exists(csv_file):  # 确保 CSV 文件存在
                df = pd.read_csv(csv_file)
                method_data[folder_name] = df  # 将数据存入字典中，键是方法名称

    all_sig_nums = df['sig_num'].unique()
    snr_range = np.sort(df['snr'].unique())
    snapshot_range = np.sort(df['snap_num'].unique())

    # 遍历所有组合情况
    for sig_num in sorted(all_sig_nums):
        for snr in snr_range:
            for snapshot in snapshot_range:
                    
                # # 固定 sig_num, snr, doa_gap，绘制 RMSE 随 snapshot 变化的图
                # plt.figure(figsize=(14, 6))
                # for method, df in method_data.items():
                #     df_filtered = df[(df['sig_num'] == sig_num) & 
                #                     (df['snr'] == snr)]
                #     if df_filtered.empty:
                #         continue
                #     sns.lineplot(data=df_filtered, x='snap_num', y='RMSE', label=method, marker='o')
                
                # plt.yscale('log')
                # # plt.title(f'RMSE vs Snapshot for Sig_Num={sig_num}, SNR={snr}')
                # plt.xlabel(r'Number of Snapshot ($\times 10^2$)', fontsize=18, fontweight='bold')
                # plt.xticks([snapshot_range], fontsize=18)
                # plt.yticks(fontsize=18)
                # plt.ylabel(r'RMSE ($^\circ$)', fontsize=18, fontweight='bold')
                # plt.ylim(0.1, 30)
                # plt.legend()
                # plt.grid(True)
                # plt.savefig(os.path.join(save_path, f'Sig_Num-{sig_num}_SNR-{snr}_RMSE_vs_Snapshot.pdf'))
                # plt.close()
                
                # 固定 sig_num, snapshot, doa_gap，绘制 RMSE 随 snr 变化的图
                plt.figure(figsize=(14, 8))
                plot_order = ["DCT-ViT", "CNN + DCT-ViT", "CNN + Effcient DCT-ViT", "MC-ViT"]
                method_markers = {
                    "DCT-ViT":"o",
                    "CNN + DCT-ViT":"s",
                    "CNN + Effcient DCT-ViT":"D",
                    "MC-ViT":"^"
                }
                for method in plot_order:
                    if method not in method_data:
                        continue
                    df = method_data[method]
                    df_filtered = df[(df['sig_num'] == sig_num) & (df['snap_num'] == snapshot)]
                    if df_filtered.empty:
                        continue
                    sns.lineplot(
                        data=df_filtered,
                        x='snr',
                        y='RMSE',
                        label=method,
                        marker=method_markers.get(method, 'o'),  # 默认 marker 为 'o'
                        linewidth=3.5,
                        linestyle='--',
                        markersize=16
                    )

                plt.yscale('log')
                # plt.title(f'RMSE vs SNR for Sig_Num={sig_num}, Snapshot={snapshot}')
                plt.xlabel('SNR(dB)', fontsize=24, fontweight='bold')
                plt.xticks(snr_range, fontsize=24, fontweight='bold')
                plt.ylabel(r'RMSE ($^\circ$)', fontsize=24, fontweight='bold')
                plt.yticks(fontsize=24, fontweight='bold')
                plt.ylim(0.1, 30)
                plt.legend(
                    title="Method",        # 图例标题
                    fontsize=20,           # 图例中项目的字体大小
                    title_fontsize=20,     # 图例标题的字体大小
                    # prop = {'weight': 'bold'}
                )
                plt.grid(True)
                plt.savefig(os.path.join(save_path, f'Sig_Num-{sig_num}_Snapshot-{snapshot}_RMSE_vs_SNR.pdf'))
                plt.close()

## 主函数
if __name__ == '__main__':
    plot_results_from_folders('/home/yxxie/Documents/mutli_cls_tran_cnn/fig2_Ablation/val_result')