import os
import tqdm
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
from catboost import CatBoostRanker, Pool
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean, sqeuclidean
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def load_data(train_path, test_path):
    """Загружает тренировочные и тестовые данные."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_text(text):
    """Простая предобработка текста для BM25 и совпадений."""
    if pd.isna(text):
        return []

    tokens = re.findall(r'\w+', text.lower())
    return tokens


def create_text_features(df):
    """Создаёт текстовые признаки."""
    df['combined_product_text'] = (
            df['product_title'].fillna('') + ' ' +
            df['product_description'].fillna('') + ' ' +
            df['product_bullet_point'].fillna('') + ' ' +
            df['product_brand'].fillna('') + ' ' +
            df['product_color'].fillna('')
    )
    df['full_text'] = df['combined_product_text']  # Для кросс-энкодера
    return df


def build_candidate_pool(df, top_n=100):
    """Строит кандидатную модель (BM25) для каждого query_id и возвращает DataFrame с id и bm25_score."""
    print("Создание кандидатной модели BM25...")
    unique_queries = df[['query_id', 'query']].drop_duplicates()
    candidate_pools = []
    for _, q_row in unique_queries.iterrows():
        query_id = q_row['query_id']
        query_text = q_row['query']

        query_subset = df[df['query_id'] == query_id].copy()
        if query_subset.empty:
            print(f"Предупреждение: Нет данных для query_id {query_id}.")
            continue

        tokenized_corpus = [preprocess_text(doc) for doc in query_subset['combined_product_text']]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = preprocess_text(query_text)
        doc_scores = bm25.get_scores(tokenized_query)

        top_n_indices = np.argsort(doc_scores)[::-1][:top_n]

        top_ids = query_subset.iloc[top_n_indices]['id'].values
        top_scores = doc_scores[top_n_indices]

        candidate_pools.append(pd.DataFrame({
            'id': top_ids,
            'bm25_score': top_scores
        }))

    candidate_pool_df = pd.concat(candidate_pools, ignore_index=True)
    return candidate_pool_df


def get_cross_encoder_scores(df, tokenizer, model, device, batch_size=16):
    """Получает предсказания кросс-энкодера."""
    print("Получение предсказаний кросс-энкодера...")
    texts = list(zip(df['query'], df['combined_product_text']))
    scores = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
        scores.extend(batch_scores)
    return scores


def get_sentence_embeddings(df, embedder_model):
    """Получает эмбеддинги для запросов и продуктов."""
    print("Получение эмбеддингов...")
    queries = embedder_model.encode(df['query'].tolist(), show_progress_bar=True)
    products = embedder_model.encode(df['combined_product_text'].tolist(), show_progress_bar=True)
    return queries, products


def calculate_embedding_features(queries, products):
    """Вычисляет признаки на основе эмбеддингов."""
    print("Вычисление признаков на основе эмбеддингов...")
    features = []
    for q_emb, p_emb in zip(queries, products):
        cos_sim = 1 - cosine(q_emb, p_emb)
        l2_dist = euclidean(q_emb, p_emb)
        dot_prod = np.dot(q_emb, p_emb)
        features.append([cos_sim, l2_dist, dot_prod])
    return np.array(features)


def calculate_text_features(df):
    """Вычисляет простые текстовые признаки."""
    print("Вычисление текстовых признаков...")
    features = []
    for _, row in df.iterrows():
        q_len = len(row['query'])
        p_title_len = len(row['product_title']) if pd.notna(row['product_title']) else 0
        p_desc_len = len(row['product_description']) if pd.notna(row['product_description']) else 0
        full_text_len = len(row['combined_product_text'])

        q_tokens = set(preprocess_text(row['query']))
        p_tokens = set(preprocess_text(row['combined_product_text']))
        common_tokens = len(q_tokens.intersection(p_tokens))

        features.append([q_len, p_title_len, p_desc_len, full_text_len, common_tokens])
    return np.array(features)


def main():

    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    submission_dir = 'result'
    submission_path = os.path.join(submission_dir, 'submission.csv')

    print("Загрузка данных...")
    train_df, test_df = load_data(train_path, test_path)

    print("Подготовка текстовых признаков...")
    train_df = create_text_features(train_df)
    test_df = create_text_features(test_df)

    print("Загрузка кросс-энкодера...")
    ce_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ce_tokenizer = AutoTokenizer.from_pretrained(ce_model_name)
    ce_model = AutoModelForSequenceClassification.from_pretrained(ce_model_name)
    ce_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ce_model.to(ce_device)
    ce_model.eval()

    print("Загрузка эмбеддинговой модели...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Отбор кандидатов BM25 для train...")
    train_candidate_pool_df = build_candidate_pool(train_df, top_n=100)
    filtered_train_df = train_df[train_df['id'].isin(train_candidate_pool_df['id'])].copy()
    filtered_train_df = filtered_train_df.merge(train_candidate_pool_df[['id', 'bm25_score']], on='id', how='left')
    print(f"Отфильтрованный train размер: {len(filtered_train_df)}")

    print("Отбор кандидатов BM25 для test...")
    test_candidate_pool_df = build_candidate_pool(test_df, top_n=100)
    filtered_test_df = test_df[test_df['id'].isin(test_candidate_pool_df['id'])].copy()
    filtered_test_df = filtered_test_df.merge(test_candidate_pool_df[['id', 'bm25_score']], on='id', how='left')
    print(f"Отфильтрованный test размер: {len(filtered_test_df)}")

    ce_scores_train = get_cross_encoder_scores(filtered_train_df, ce_tokenizer, ce_model, ce_device)
    queries_emb_train, products_emb_train = get_sentence_embeddings(filtered_train_df, st_model)
    emb_features_train = calculate_embedding_features(queries_emb_train, products_emb_train)
    text_features_train = calculate_text_features(filtered_train_df)

    X_train = np.hstack([
        emb_features_train,
        text_features_train,
        np.array(ce_scores_train).reshape(-1, 1),
        filtered_train_df[['bm25_score']].values
    ])
    y_train = filtered_train_df['relevance'].values
    group_id_train = filtered_train_df['query_id'].values

    ce_scores_test = get_cross_encoder_scores(filtered_test_df, ce_tokenizer, ce_model, ce_device)
    queries_emb_test, products_emb_test = get_sentence_embeddings(filtered_test_df, st_model)
    emb_features_test = calculate_embedding_features(queries_emb_test, products_emb_test)
    text_features_test = calculate_text_features(filtered_test_df)

    X_test = np.hstack([
        emb_features_test,
        text_features_test,
        np.array(ce_scores_test).reshape(-1, 1),
        filtered_test_df[['bm25_score']].values  # Добавляем BM25 как признак
    ])
    group_id_test = filtered_test_df['query_id'].values

    print("Обучение CatBoostRanker...")
    train_pool = Pool(data=X_train, label=y_train, group_id=group_id_train)

    model = CatBoostRanker(
        iterations=200,
        learning_rate=0.05,
        depth=8,
        loss_function='YetiRank',
        eval_metric='NDCG',
        verbose=50
    )
    model.fit(train_pool)

    print("Предсказание на тесте...")
    test_predictions = model.predict(X_test)

    print("Создание submission файла...")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'prediction': 0.0
    })

    prediction_map = dict(zip(filtered_test_df['id'], test_predictions))
    submission_df['prediction'] = submission_df['id'].map(prediction_map).fillna(0.0)

    os.makedirs(submission_dir, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Файл submission сохранён в {submission_path}")


if __name__ == "__main__":
    main()