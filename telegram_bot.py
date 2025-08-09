import io
import os
import requests
import logging
from typing import Dict
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import io
import asyncio

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get bot token
TOKEN = os.getenv('TELEGRAM_TOKEN')

if not TOKEN:
    logger.error("❌ TELEGRAM_TOKEN environment variable not found!")
    exit(1)

# Use a simple heuristic-based AI detection instead of heavy ML models
def simple_ai_detection(image):
    """
    Simple AI detection using image characteristics
    This is a lightweight alternative to heavy ML models
    """
    # Get image properties
    width, height = image.size
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get some basic statistics
    pixels = list(image.getdata())
    total_pixels = len(pixels)
    
    # Calculate some heuristics (this is simplified)
    # Real AI detection would use actual ML models
    
    # Check for perfect symmetry (common in AI art)
    symmetry_score = 0
    
    # Check for unusual color distributions
    red_values = [p[0] for p in pixels[:1000]]  # Sample first 1000 pixels
    green_values = [p[1] for p in pixels[:1000]]
    blue_values = [p[2] for p in pixels[:1000]]
    
    # Simple heuristics
    avg_red = sum(red_values) / len(red_values)
    avg_green = sum(green_values) / len(green_values)
    avg_blue = sum(blue_values) / len(blue_values)
    
    # Very basic AI probability calculation
    # This is a placeholder - real detection needs actual ML models
    color_variance = abs(avg_red - avg_green) + abs(avg_green - avg_blue)
    aspect_ratio = width / height
    
    # Simple scoring (not accurate, just for demo)
    ai_probability = min(0.9, max(0.1, (color_variance / 100 + abs(aspect_ratio - 1)) * 0.3))
    
    return {
        'ai_probability': ai_probability,
        'real_probability': 1 - ai_probability,
        'confidence': 'Low'  # Always low for this simple method
    }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_msg = """
🤖 **Welcome to Thibitisha AI Detection Bot!**

⚠️ **Note**: Currently using lightweight detection for better performance.
For more accurate results, try the full version.

📋 **Commands:**
/start - Show this welcome message
/info - Bot information
/status - Check bot status

📸 **Usage:**
Just send me any image and I'll analyze it!

🔧 **Powered by:** Simple Image Analysis
"""
    
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Info command handler"""
    info_msg = """
ℹ️ **Bot Information**

🎯 **Purpose:** AI-Generated Image Detection
📊 **Method:** Lightweight Image Analysis
💾 **Memory Usage:** Optimized for 512MB
🔧 **Framework:** Basic Python + PIL

📋 **Supported Formats:**
• JPEG/JPG, PNG, WebP
• GIF (static), BMP, TIFF
• Max size: 20MB

⚠️ **Accuracy Note:**
This is a lightweight version with basic detection.
Results are approximate and for demonstration purposes.

⚡ **Response Time:** 1-2 seconds per image
🔒 **Privacy:** Images are not stored
"""
    
    await update.message.reply_text(info_msg, parse_mode='Markdown')

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Status command handler"""
    status_msg = """
📊 **Bot Status Report**

🤖 **Bot:** 🟢 Online and Ready
🧠 **Detection:** 🟢 Lightweight Mode Active
📡 **Connection:** 🟢 Connected to Telegram
💾 **Memory:** 🟢 Optimized for 512MB

⚠️ **Mode:** Lightweight Detection
📊 **Accuracy:** Basic/Demo Level
🕐 **Last Update:** Just now
"""
    
    await update.message.reply_text(status_msg, parse_mode='Markdown')

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads and perform simple AI detection"""
    
    try:
        # Send initial processing message
        processing_msg = await update.message.reply_text("📥 **Downloading image...**", parse_mode='Markdown')
        
        # Get the photo
        photo = update.message.photo[-1]
        
        # Download the photo
        file = await context.bot.get_file(photo.file_id)
        image_data = await file.download_as_bytearray()
        
        # Update status
        await processing_msg.edit_text("🧠 **Analyzing image...**", parse_mode='Markdown')
        
        # Load image
        image = Image.open(io.BytesIO(image_data))
        
        # Resize if too large
        if max(image.size) > 512:
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        
        # Run simple detection
        result = simple_ai_detection(image)
        
        # Format results
        ai_prob = result['ai_probability'] * 100
        real_prob = result['real_probability'] * 100
        
        if ai_prob > 60:
            verdict = "⚠️ Possibly AI-Generated"
            emoji = "🔴"
        elif ai_prob < 40:
            verdict = "✅ Likely Real Photo"  
            emoji = "🟢"
        else:
            verdict = "🤔 Uncertain"
            emoji = "🟡"
        
        result_msg = f"""{emoji} **AI Detection Results**

**Verdict:** {verdict}
**AI Probability:** {ai_prob:.1f}%
**Confidence:** {result['confidence']}

📊 **Breakdown:**
• Real Photo: {real_prob:.1f}%
• AI-Generated: {ai_prob:.1f}%

⚠️ **Note:** Lightweight detection mode
🔧 **Method:** Basic image analysis

For more accurate results, use the full ML version.
"""
        
        await processing_msg.edit_text(result_msg, parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"""
❌ **Processing Error**

Sorry, I couldn't analyze this image.

**Error:** {str(e)[:100]}...

Please try again with a different image.
"""
        
        await processing_msg.edit_text(error_msg, parse_mode='Markdown')
        logger.error(f"Error processing image: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    response = """
📸 **Send me an image to analyze!**

I'll use lightweight detection to analyze if your image might be AI-generated.

⚠️ **Note:** Currently in demo mode with basic detection.
Results are approximate.

Use /info for more details.
"""
    
    await update.message.reply_text(response, parse_mode='Markdown')

def main():
    """Main function"""
    logger.info("🚀 Starting Thibitisha AI Detection Bot (Lightweight Mode)...")
    
    # Create application
    application = ApplicationBuilder().token(TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("info", info))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("✅ Bot handlers registered!")
    logger.info("🎯 Bot is now running in lightweight mode!")
    
    # Start the bot
    application.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()