# Vector database

Listen --> https://youtu.be/7xTaF8MTrts

Vector Database คือระบบจัดเก็บและค้นหาข้อมูลที่ถูกออกแบบมาสำหรับ "เวกเตอร์ความหมาย" (Vector Embeddings) โดยเฉพาะ ซึ่งรองรับการค้นหาแบบความคล้ายคลึง (Similarity Search) ได้อย่างมีประสิทธิภาพในงานขนาดใหญ่

## Vector Embeddings คืออะไร

คือการแปลงข้อมูล เช่น ข้อความ รูปภาพ เสียง ฯลฯ ให้เป็นตัวเลขหลายมิติ (High-dimensional Vectors) โมเดลฝึกมาเพื่อให้เวกเตอร์นี้เก็บ "ความหมาย" ของข้อมูลไว้ภายใน โดยความหมายจะถูกเข้ารหัสอยู่ใน ทิศทาง (Angle) มากกว่าความยาวของเวกเตอร์

จึงสามารถนำไปใช้วัดความคล้ายของข้อมูลต่าง ๆ ได้ เช่น
• ค้นหาข้อความที่ใกล้เคียงกัน
• จัดกลุ่มข้อมูล (Clustering)
• ทำระบบแนะนำ (Recommendation)
• วิเคราะห์ภาพ/เสียงเชิงความหมาย

## การวัดความคล้าย (Distance / Similarity Metrics)

เมื่อใช้งานจริง เช่น Inference หรือ Search จะใช้ตัววัดระยะเพื่อตรวจหาข้อมูลที่คล้ายกันที่สุด โดยหลัก ๆ มีสองแบบที่ใช้บ่อย

• Cosine Similarity

เหมาะกับงาน Search เป็นหลัก เพราะมองเฉพาะ "มุม" ระหว่างเวกเตอร์ โดยไม่สนความยาว เวกเตอร์ที่มีความหมายเหมือนกันมักชี้ไปในทิศทางคล้ายกัน

• Euclidean Distance

เหมาะกับงานจัดกลุ่มข้อมูล เช่น K-Means เพราะต้องวัด "ระยะทางจริง" ระหว่างจุดข้อมูลบนพื้นที่เวกเตอร์

## แนวทางปฏิบัติจริงสำหรับระบบ Search / RAG

เลือก Cosine Similarity เป็นตัวตั้งต้น ปลอดภัยและได้ผลดีที่สุดในงานทั่วไป

ถ้าใช้ Vector Database (Pinecone, Weaviate, Milvus ฯลฯ) และ Embedding เป็น Normalized Vector อยู่แล้ว 
→ เลือกใช้ Dot Product แทน Cosine ได้เลย เพราะ
 • เร็วกว่า
 • ให้อันดับผลลัพธ์เหมือน Cosine เป๊ะ

→ ทำไมควรใช้ Dot Product แทนที่ Cosine  
เพราะเมื่อเวกเตอร์ทุกตัวถูก Normalize แล้วจะมีความยาว = 1 ซึ่งทำให้ค่า Dot Product = Cosine ดังสมการ

```equation
cosine(a, b) = dot(a, b) / (|a| |b|)  = dot(a, b)
```  

การคำนวณ Dot Product เร็วกว่าการคำนวณ Cosine ทำให้ประหยัดทรัพยากรของระบบ search ขนาดใหญ่

หมายเหตุ: เรื่องการ Training Model

ระหว่าง Training อาจใช้ MSE (Mean Squared Error) หรือ Loss แบบ L2 เป็นตัวสอนโมเดลเพื่อบอกว่า
"เวกเตอร์นี้ควรอยู่ตำแหน่งประมาณนี้ ถ้าผิดก็ปรับค่าจนเข้าใกล้ความจริง"

เหตุผลที่ MSE นิยมใช้ใน Machine Learning:
• สมการง่ายต่อการหา Gradient
• การยกกำลังสองทำให้ Error ใหญ่ ๆ ถูกลงโทษหนักกว่า เช่น
 ผิด 2 หน่วย → 2² = 4
 ผิด 10 หน่วย → 10² = 100

## สรุปสั้นๆ

• Vector DB = ฐานข้อมูลสำหรับเก็บและค้นหา Vector Embeddings
• Search ส่วนใหญ่ใช้ Cosine
• Clustering ใช้ Euclidean
• ถ้า Embedding ถูก Normalize → ใช้ Dot Product เพื่อให้เร็วขึ้น
• Training มักใช้ MSE เพราะปรับ Gradient ง่ายและลงโทษ Error ใหญ่ได้ดี

## Multimodal

คือการจัดเก็บข้อมูลที่หลากหลายรูปแบบ เช่น ข้อความ รูปภาพ เสียง วิดีโอ ฯลฯ ในฐานข้อมูลเดียวกัน ทำให้สามารถค้นหาและวิเคราะห์ข้อมูลจากหลายแหล่งได้อย่างมีประสิทธิภาพ

เราสามารถค้นหาข้อมูลที่เกี่ยวข้องกับคำค้นหาที่เป็นข้อความและแสดงผลลัพธ์ที่ประกอบด้วยรูปภาพหรือวิดีโอที่เกี่ยวข้องได้ ตัวอย่างเช่น  
Text-to-Image Search, Image-to-Text Search, Image-to-Image Search เป็นต้น

### Mutlimodal - OpenCLIP

OpenCLIP รองรับทั้ง "ข้อความ" และ "ภาพ" และนสามารถแปลข้ามกันได้อีกด้วย
Model นี้จะสร้างเวกเตอร์ที่มีความหมายร่วมกันสำหรับข้อความและรูปภาพ
แปลง่ายๆได้ว่า *เวกเตอร์ของ "คำว่าสุนัข" จะถูกวางไว้ใกล้กับเวกเตอร์ของ "รูปภาพสุนัข"* นั่นเอง

## References

- [Try Chroma](https://www.trychroma.com/)
- [Chroma Embeddings - Default All MiniLM L6 v2](https://docs.trychroma.com/docs/embeddings/embedding-functions#default-all-minilm-l6-v2)
- [Chroma - Adding Multimodal Data and Data Loaders](https://docs.trychroma.com/docs/embeddings/multimodal#adding-multimodal-data-and-data-loaders)
- [Chroma Multimodal - OpenCLIP Model](https://github.com/mlfoundations/open_clip)
- [Chroma docker image](https://hub.docker.com/r/chromadb/chroma)
- [vector db / sqlitevec / chromadb](https://mikelopster.dev/posts/vector-db-chromadb)
- [Vector Database Fundamentals](https://youtu.be/jGPxr0Qk-Vs?si=SD4bKoPRA9QtUJ9V)
