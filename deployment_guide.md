
# 🏠 نشر النموذج محلياً باستخدام Ollama

## 1. تثبيت Ollama:
```bash
# على Windows/Mac/Linux
curl -fsSL https://ollama.ai/install.sh | sh
```

## 2. إنشاء النموذج:
```bash
cd /path/to/your/model/directory
ollama create my-llama3-finetuned -f Modelfile
```

## 3. تشغيل النموذج:
```bash
ollama run my-llama3-finetuned
```

## 4. اختبار النموذج:
```bash
ollama run my-llama3-finetuned "What is artificial intelligence?"
```

## 5. استخدام عبر API:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "my-llama3-finetuned",
  "prompt": "Explain machine learning in simple terms."
}'
```

## 📊 معلومات النموذج:
- النوع: LLAMA3 Fine-tuned
- الحجم: ~4-5 GB (Q4_K_M)
- الذاكرة المطلوبة: 6-8 GB RAM
- المعالج: يفضل GPU لكن يعمل على CPU
