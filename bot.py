import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate 
)
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
load_dotenv()

Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"))

QA_SYSTEM_PROMPT = """Sen "Oqıw Orayı"nıń jasalma intellektke tiykarlanǵan, sıpayı hám júdá járdemshi assistentiseń.

QATAŃ QAǴIYDALAR:
1. **HÁRDAYIM TEK ǴANA QARAQALPAQ TILINDE LATIN JAZIWINDA JUWAP BER.** Hesh qanday kirill (Иә, да, нет), inglis (yes, no), rus yaki ózbek sózlerin qollanba. Hátte bir sóz de qospa. Máselen: "Ha" (kirill "Иә" emes), "Joq" (kirill "Нет" emes).

2. **ALDIŃǴI SÁWBET TARIYXIN HÁRDAYIM ESAPQA AL.** Eger paydalanıwshı "onıń bahası qansha?", "ol kurs", "usı kurs", "sabaqları qaysı kúni?" sıyaqlı silteme sózlerin qollansa, aldınǵı sorawlardan qaysı kurs haqqında sóz bolıp atırǵanın anıq túsiniw kerek.

MÁSELEN:
- Eger aldınǵı sorawda "Frontend kursı bar ma?" soralsa
- Keyingi "Onıń bahası qansha?" sorawı Frontend kursınıń bahası haqqında
- Juwap: "Frontend Dástúrlew kursınıń bahası ayına 1,200,000 so'm."

3. Barlıq juwaplarıńdı **tek ǵana berilgen Kontekst maǵlıwmatlarına** tiykarla. Kontekstten sırtqa shıqpa.

4. Eger sorawǵa juwap Kontekstte joq bolsa yaki soraw "Oqıw Orayı"ǵa qatıssız bolsa, "Keshiriń, bul soraw boyınsha mende anıq maǵlıwmat joq. Basqa sorawıńız bar ma yaki administrator menen baylanıw ushın +998 XX XXX XX XX nomerine qońıraw etseńiz boladı." dep juwap ber.

5. Juwaplarıń qısqa, anıq hám logikalıq bolsın. Paydalanıwshı menen hárdayım "Siz" dep sóyles.

Kontekst maǵlıwmatları:
{context_str}

Soraw: {query_str}

Juwap:"""

chat_engines = {}
conversation_history = {}

def get_or_create_chat_engine(chat_id: int):
    if chat_id not in chat_engines:
        print(f"Paydalanıwshı {chat_id} ushın jańa Chat Engine jaratılıp atır...")
        
        memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        
        qa_template = PromptTemplate(QA_SYSTEM_PROMPT)
        
        try:
            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                verbose=True,
                similarity_top_k=5,
                system_prompt=QA_SYSTEM_PROMPT
            )
            print("Context chat rejimi menen system prompt ornatıldı")
        except Exception as e:
            print(f"Context mode islemedi: {e}")
            try:
                chat_engine = index.as_chat_engine(
                    chat_mode="condense_question",
                    memory=memory,
                    verbose=True,
                    similarity_top_k=5
                )
                
                chat_engine._query_engine._response_synthesizer.update_prompts(
                    {"text_qa_template": qa_template}
                )
                print("Condense question mode bilan system prompt o'rnatildi")
            except Exception as e2:
                print(f"Condense question mode ham ishlamadi: {e2}")
                # Final fallback to simple mode
                chat_engine = index.as_chat_engine(
                    chat_mode="simple",
                    memory=memory,
                    verbose=True,
                    similarity_top_k=5
                )
                print("Simple mode isletilip atir...")
        
        chat_engines[chat_id] = chat_engine
    
    return chat_engines[chat_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id
    if chat_id in chat_engines:
        del chat_engines[chat_id]
        print(f"Paydalanıwshı {chat_id} ushın sáwbet tariyxı tazalandı.")
    if chat_id in conversation_history:
        del conversation_history[chat_id]
        print(f"Paydalanıwshı {chat_id} ushın manual history tazalandı.")
    await update.message.reply_text("Assalomu aleykum! Sáwbetimiz jańalandı. Maǵan sorawıńızdı jollasańız boladı.")


def add_to_conversation_history(chat_id: int, question: str, answer: str):
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
    conversation_history[chat_id].append({
        "question": question,
        "answer": answer
    })
    
    if len(conversation_history[chat_id]) > 5:
        conversation_history[chat_id] = conversation_history[chat_id][-5:]

def get_conversation_context(chat_id: int) -> str:
    if chat_id not in conversation_history or not conversation_history[chat_id]:
        return ""
    
    context_parts = ["ALDIŃǴI SÁWBET TARIYXI:"]
    for i, pair in enumerate(conversation_history[chat_id]):
        context_parts.append(f"Soraw {i+1}: {pair['question']}")
        context_parts.append(f"Juwap {i+1}: {pair['answer']}")
    
    context_parts.append("HÁZIRGI SORAW:")
    return "\n".join(context_parts)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_question = update.message.text
    chat_id = update.message.chat_id
    logging.info(f"Paydalanıwshı {chat_id} dan soraw keldi: {user_question}")

    try:
        conversation_context = get_conversation_context(chat_id)
        
        # Create enhanced question with context
        if conversation_context:
            enhanced_question = f"{conversation_context}\n{user_question}"
            logging.info(f"Keńeytilgen kontekstli soraw: {enhanced_question[:200]}...")
        else:
            enhanced_question = user_question
        
        chat_engine = get_or_create_chat_engine(chat_id)
        
        await update.message.reply_chat_action(action='typing')
        
        # Send enhanced question with conversation history
        response = await chat_engine.achat(enhanced_question)
        answer = str(response)
        
        add_to_conversation_history(chat_id, user_question, answer)
        
        logging.info(f"Bot juwabı: {answer[:100]}...")
        logging.info(f"Sóylesiwler tariyxı uzınlıǵı: {len(conversation_history.get(chat_id, []))}")
        
        await update.message.reply_text(answer)
        
    except Exception as e:
        logging.error(f"Qáte júz berdi: {str(e)}")
        
        if "ResourceExhausted" in str(e) or "429" in str(e) or "quota" in str(e).lower():
            error_response = "Keshiriń, házirgi waqıtta texnikalıq qıyınshılıqlar bar. Birneshe minutttan soń qayta urinip kóriń yaki administrator menen baylanıw ushın +998 XX XXX XX XX nomerine qońıraw etseńiz boladı."
        else:
            error_response = "Keshiriń, texnikalıq qıyınshılıq yuz berdi. Qayta urinip kóriń yaki administrator menen baylanıw ushın +998 XX XXX XX XX nomerine qońıraw etseńiz boladı."
        
        await update.message.reply_text(error_response)

def main() -> None:
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logging.info("Bot iske tu'sti...")
    application.run_polling()

if __name__ == "__main__":
    main()