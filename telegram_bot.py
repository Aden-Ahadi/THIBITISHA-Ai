import os
import requests
import logging
from typing import Dict, Optional
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline
from PIL import Image
import asyncio

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Manually load .env file
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = value

# Bot configuration
BOT_TOKEN =os.environ.get("TELEGRAM_TOKEN")

class HuggingFaceAIDetector:
    def __init__(self):
        self.classifier = None
        self.model_name = None
        self.load_model()
    
    def load_model(self):
        """Try to load AI detection models with fallback options"""
        models_to_try = [
            "umm-maybe/AI-image-detector",
            "Organika/sdxl-detector", 
            "saltacc/anime-ai-detect",
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying to load model: {model_name}")
                
                self.classifier = pipeline(
                    "image-classification",
                    model=model_name,
                    trust_remote_code=True,
                    use_fast=True
                )
                
                self.model_name = model_name
                logger.info(f"✅ Successfully loaded model: {model_name}")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.error("❌ All models failed to load. Bot will use dummy responses.")
        return False
    
    def classify_image(self, image_path: str) -> Dict[str, float]:
        """Classify if an image is AI-generated"""
        if self.classifier is None:
            logger.warning("No model available - returning dummy result")
            return {
                "ai_probability": 0.5, 
                "error": "AI detection model not available",
                "model_used": "none"
            }
            
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Resize if too large (helps with memory and speed)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            logger.info(f"Processing image with model: {self.model_name}")
            
            # Run classification
            results = self.classifier(image)
            
            # Extract AI probability based on model type
            ai_probability = self._extract_ai_probability(results)
            
            logger.info(f"Classification results: {results}")
            logger.info(f"Final AI probability: {ai_probability:.3f}")
            
            return {
                "ai_probability": float(ai_probability),
                "model_used": self.model_name,
                "raw_results": results
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                "ai_probability": 0.5, 
                "error": str(e),
                "model_used": self.model_name
            }
    
    def _extract_ai_probability(self, results) -> float:
        """Extract AI probability from classification results"""
        if not results:
            return 0.5
        
        # Handle different model output formats
        ai_keywords = ['artificial', 'ai', 'generated', 'fake', 'synthetic', 'artificial intelligence']
        real_keywords = ['real', 'human', 'authentic', 'natural', 'photo', 'photograph']
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            # Check for AI-related labels
            if any(keyword in label for keyword in ai_keywords):
                return score
            elif any(keyword in label for keyword in real_keywords):
                return 1.0 - score
        
        # For binary classification, assume first result is more likely
        if len(results) >= 2:
            return results[0]['score']
        
        return 0.5

# Initialize detector
detector = HuggingFaceAIDetector()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_status = "🟢 Ready" if detector.classifier else "🔴 Limited (No AI model)"
    
    welcome_text = (
        "🤖 **AI Image Detection Bot**\n\n"
        "Send me any image and I'll detect if it's AI-generated!\n\n"
        f"**Status:** {model_status}\n"
        f"**Model:** {detector.model_name or 'None'}\n\n"
        "📸 Just send me an image to get started!\n"
        "💡 Use /info for more details"
    )
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    image_path = None
    try:
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Download image
        photo = update.message.photo[-1]  
        file = await context.bot.get_file(photo.file_id)
        image_path = f"temp_{photo.file_id}.jpg"
        
        await update.message.reply_text("📥 Downloading image...")
        
        # Download with timeout
        response = requests.get(file.file_path, timeout=30)
        response.raise_for_status()
        
        # Save image
        with open(image_path, "wb") as f:
            f.write(response.content)
        
        await update.message.reply_text("🧠 Running AI detection...")
        
        # Classify image
        result = detector.classify_image(image_path)
        
        # Handle errors
        if "error" in result:
            error_msg = (
                f"❌ **Detection Error**\n\n"
                f"Error: {result['error']}\n\n"
                f"This might be due to:\n"
                f"• Network connectivity issues\n"
                f"• Model loading problems\n"
                f"• Image format issues\n\n"
                f"Please try again later or with a different image."
            )
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            return
            
        ai_probability = result["ai_probability"]
        
        # Determine verdict with confidence levels
        if ai_probability > 0.85:
            verdict = "🤖 **VERY LIKELY AI-Generated**"
            confidence = "Very High"
            color = "🔴"
        elif ai_probability > 0.7:
            verdict = "⚠️ **Probably AI-Generated**"
            confidence = "High"
            color = "🟠"
        elif ai_probability > 0.55:
            verdict = "🤔 **Possibly AI-Generated**"
            confidence = "Medium"
            color = "🟡"
        elif ai_probability > 0.3:
            verdict = "📸 **Probably Real Photo**"
            confidence = "Medium"
            color = "🟢"
        else:
            verdict = "✅ **Very Likely Real Photo**"
            confidence = "High"
            color = "🟢"
        
        # Create detailed response
        response_text = (
            f"{color} **AI Detection Results**\n\n"
            f"**Verdict:** {verdict}\n"
            f"**AI Probability:** {ai_probability:.1%}\n"
            f"**Confidence:** {confidence}\n\n"
            f"📊 **Breakdown:**\n"
            f"• Real Photo: {(1-ai_probability):.1%}\n"
            f"• AI-Generated: {ai_probability:.1%}\n\n"
            f"🔧 **Model:** {result.get('model_used', 'Unknown')}\n"
          
        )
        
        await update.message.reply_text(response_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        await update.message.reply_text(
            "❌ **Processing Error**\n\n"
            "Failed to analyze the image. Please try:\n"
            "• A different image format (JPG/PNG)\n"
            "• A smaller file size\n"
            "• Waiting a moment and trying again"
        )
        
    finally:
        # Clean up temporary file
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📷 **Send me an image to analyze!**\n\n"
        "I can detect AI-generated images from:\n"
        "• DALL-E, Midjourney, Stable Diffusion\n"
        "• Modern AI art generators\n"
        "• Digital art vs photographs\n\n"
        "💡 Tip: Higher resolution images work better!"
    )

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_info = f"**Model:** {detector.model_name}\n**Status:** {'🟢 Active' if detector.classifier else '🔴 Unavailable'}" if detector.model_name else "**Model:** Not loaded\n**Status:** 🔴 Limited functionality"
    
    info_text = (
        "ℹ️ **Bot Information**\n\n"
        f"{model_info}\n"
        "**Platform:** Hugging Face 🤗\n\n"
        "**What I can detect:**\n"
        "• AI-generated artwork\n"
        "• Synthetic images\n"
        "• Digital art vs photos\n"
        "• Modern AI model outputs\n\n"
        "**Tips for best results:**\n"
        "• Use high-resolution images\n"
        "• Avoid heavily compressed images\n"
        "• Clear, unambiguous images work best\n\n"
        "**Limitations:**\n"
        "• Results are not 100% accurate\n"
        "• May struggle with edge cases\n"
        "• Performance depends on model availability"
    )
    await update.message.reply_text(info_text, parse_mode='Markdown')

def main():
    logger.info("🚀 Starting AI Detection Bot...")
    
    try:
        # Create application
        app = (
            ApplicationBuilder()
            .token(BOT_TOKEN)
            .read_timeout(60)
            .write_timeout(60)
            .connect_timeout(60)
            .pool_timeout(60)
            .build()
        )
        
        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("info", info_command))
        app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        app.add_handler(MessageHandler(~filters.PHOTO, handle_non_photo))
        
        logger.info("✅ Bot handlers registered!")
        logger.info("Press Ctrl+C to stop")
        
        # Start polling
        app.run_polling(
            drop_pending_updates=True,
            timeout=30,
            bootstrap_retries=3
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Bot stopped. Please check the error and restart manually.")

if __name__ == "__main__":
    main()