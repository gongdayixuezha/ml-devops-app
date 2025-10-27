import os
import requests
import tarfile  # 替换zipfile，用于处理tar.gz文件
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 配置路径
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

def download_imdb_data():
    """自动下载IMDb数据集（50k评论，tar.gz格式）并解压"""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = os.path.join(DATA_RAW_PATH, "aclImdb_v1.tar.gz")
    
    # 检查文件是否已下载，若损坏则重新下载
    if not os.path.exists(save_path) or os.path.getsize(save_path) < 80000000:  # 数据集约80MB
        print("Downloading IMDb data (80MB)...")
        # 流式下载，避免文件损坏
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()  # 检查下载是否成功
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download completed.")
    
    # 解压tar.gz文件（核心修正：替换zipfile为tarfile）
    extract_path = os.path.join(DATA_RAW_PATH, "aclImdb")
    if not os.path.exists(extract_path):
        print("Extracting data...")
        with tarfile.open(save_path, "r:gz") as tar_ref:  # 处理tar.gz格式
            tar_ref.extractall(DATA_RAW_PATH)
        print("Data extracted.")
    return extract_path

def load_and_preprocess_data(extract_path, version="v1"):
    """加载并预处理数据，生成v1/v2版本"""
    # 加载正负样本
    def load_reviews(folder):
        reviews = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                    reviews.append(f.read())
        return reviews
    
    pos_reviews = load_reviews(os.path.join(extract_path, "train", "pos")) + load_reviews(os.path.join(extract_path, "test", "pos"))
    neg_reviews = load_reviews(os.path.join(extract_path, "train", "neg")) + load_reviews(os.path.join(extract_path, "test", "neg"))
    
    # 构建DataFrame
    data = pd.DataFrame({
        "text": pos_reviews + neg_reviews,
        "label": [1]*len(pos_reviews) + [0]*len(neg_reviews)  # 1=正面，0=负面
    }).sample(frac=1, random_state=42)  # 打乱数据

    # 预处理：v1（基础版） vs v2（优化版）
    if version == "v1":
        # v1：仅TF-IDF，不做额外清洗
        tfidf = TfidfVectorizer(max_features=5000, stop_words=None)  # 不排除停用词
    elif version == "v2":
        # v2：优化版（去除停用词、限制文本长度）
        tfidf = TfidfVectorizer(
            max_features=8000,
            stop_words="english",  # 去除英文停用词
            max_df=0.9,  # 排除高频词（出现超过90%样本的词）
            min_df=5     # 排除低频词（出现少于5次的词）
        )
        # 额外清洗：去除超短文本（少于10个字符）
        data = data[data["text"].str.len() >= 10].reset_index(drop=True)
    
    # 特征转换
    X = tfidf.fit_transform(data["text"]).toarray()
    y = data["label"].values
    
    # 分割训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 保存预处理后的数据
    save_dir = os.path.join(DATA_PROCESSED_PATH, version)
    os.makedirs(save_dir, exist_ok=True)
    pd.to_pickle((X_train, y_train), os.path.join(save_dir, "train.pkl"))
    pd.to_pickle((X_test, y_test), os.path.join(save_dir, "test.pkl"))
    pd.to_pickle(tfidf, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
    print(f"Version {version} data saved to {save_dir}")
    return save_dir

if __name__ == "__main__":
    # 1. 下载并解压数据（修正了解压逻辑）
    extract_path = download_imdb_data()
    
    # 2. 生成v1数据集（基础版）
    load_and_preprocess_data(extract_path, version="v1")
    
    # 3. 生成v2数据集（优化版）
    load_and_preprocess_data(extract_path, version="v2")
    
    # 4. 用DVC跟踪data目录
    os.system("dvc add data")  # 生成data.dvc和.data.hash
    os.system("git add data.dvc .gitignore")
    os.system('git commit -m "data: add v1 and v2 processed datasets (dvc tracked)"')
    
    # 5. 推送数据到本地远程（无Azure版本）
    os.system("dvc push -r local_remote")