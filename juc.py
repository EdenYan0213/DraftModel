import requests
import json
import concurrent.futures
from threading import Lock
import time
from collections import defaultdict

# 配置参数
# url = "http://data-rag.fp-data.shopee.sg/kb/retrieval"
headers = {'Content-Type': 'application/json'}
url = "http://localhost:8080/kb/retrieval"

# 请求载荷模板
payload_template = {
    "query": "no funds yet",
    "metaColumn": "main_question, answer,dialogue_type",
    "recallNum": 20,
    "region": "ID",
    "scenario": "chatbot",
    "agentId": 0,
    "recallInfo": [
        {
            "recallLabel": "question",
            "recallType": "similarity",
            "scoreThreshold": 0.7,
            "embeddingModel": "BAAI/bge-m3",
            "vectorName": "vector_hnsw",
            "indexName": "1_0_1",
            "quotaRatio": 0.6
        },
        {
            "recallLabel": "answer",
            "recallType": "similarity",
            "scoreThreshold": 0.7,
            "embeddingModel": "BAAI/bge-m3",
            "vectorName": "vector_hnsw",
            "indexName": "2_0_1",
            "quotaRatio": 0.4
        }
    ]
}

# 统计变量
success_count = 0
error_count = 0
error_details = defaultdict(int)  # 记录错误详情
lock = Lock()


def send_request(query_text=None, request_id=None):
    """
    发送单个请求并捕获详细错误信息
    """
    global success_count, error_count, error_details

    payload = payload_template.copy()
    if query_text:
        payload["query"] = query_text

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=10
        )

        with lock:
            if response.status_code == 200:
                success_count += 1
                return {"status": "success", "request_id": request_id, "response": response.text}
            else:
                error_count += 1
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                error_details[error_msg] += 1
                return {"status": "error", "request_id": request_id, "error": error_msg,
                        "status_code": response.status_code}

    except requests.exceptions.Timeout:
        with lock:
            error_count += 1
            error_msg = "Request Timeout"
            error_details[error_msg] += 1
        return {"status": "error", "request_id": request_id, "error": "Timeout"}

    except requests.exceptions.ConnectionError:
        with lock:
            error_count += 1
            error_msg = "Connection Error"
            error_details[error_msg] += 1
        return {"status": "error", "request_id": request_id, "error": "Connection failed"}

    except requests.exceptions.RequestException as e:
        with lock:
            error_count += 1
            error_msg = f"Request Exception: {str(e)[:100]}"
            error_details[error_msg] += 1
        return {"status": "error", "request_id": request_id, "error": str(e)}

    except Exception as e:
        with lock:
            error_count += 1
            error_msg = f"Unknown Error: {str(e)[:100]}"
            error_details[error_msg] += 1
        return {"status": "error", "request_id": request_id, "error": str(e)}


def high_concurrency_test_with_error_details(concurrent_requests=50, total_requests=200):
    """
    高并发测试函数，包含详细的错误信息统计
    """
    global success_count, error_count, error_details
    success_count = 0
    error_count = 0
    error_details.clear()

    start_time = time.time()

    print(f"开始高并发测试...")
    print(f"并发数: {concurrent_requests}, 总请求数: {total_requests}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        # 提交任务
        futures = []
        for i in range(total_requests):
            query_text = f"test query {i}"
            future = executor.submit(send_request, query_text, i)
            futures.append(future)

        # 收集结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=15)
                results.append(result)
            except Exception as e:
                with lock:
                    error_count += 1
                    error_msg = f"Future Exception: {str(e)[:100]}"
                    error_details[error_msg] += 1

    end_time = time.time()

    # 输出统计结果
    print("\n" + "=" * 50)
    print("测试结果统计:")
    print("=" * 50)
    print(f"总请求数: {total_requests}")
    print(f"成功请求数: {success_count}")
    print(f"失败请求数: {error_count}")
    print(f"成功率: {success_count / total_requests * 100:.2f}%")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"平均QPS: {total_requests / (end_time - start_time):.2f}")

    # 输出错误详情
    if error_details:
        print("\n" + "-" * 30)
        print("错误详情统计:")
        print("-" * 30)
        for error_msg, count in sorted(error_details.items(), key=lambda x: x[1], reverse=True):
            print(f"{count:4d} 次 -> {error_msg}")

    # 显示部分失败请求详情
    failed_requests = [r for r in results if r and r.get("status") == "error"]
    if failed_requests:
        print("\n" + "-" * 30)
        print("失败请求示例 (前5个):")
        print("-" * 30)
        for i, failed_req in enumerate(failed_requests[:5]):
            print(f"[请求 {failed_req.get('request_id', 'N/A')}] {failed_req.get('error', 'Unknown error')}")


# 使用示例
if __name__ == "__main__":
    # 运行高并发测试
    high_concurrency_test_with_error_details(concurrent_requests=20, total_requests=100)
