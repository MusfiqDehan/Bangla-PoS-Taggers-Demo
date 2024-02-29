import gradio as gr
from gradio_rich_textbox import RichTextbox

from helper.text_preprocess import space_punc
from helper.pos_taggers import select_pos_tagger
from helper.translators import select_translator


def bn_postagger(src, translator, tagger):
    """
    Bangla PoS Tagger
    """
    
    src = space_punc(src)

    tgt_base, tgt = select_translator(src, translator)

    result, pos_accuracy = select_pos_tagger(src, tgt, tagger)

    return tgt_base, result, pos_accuracy
    

# Define the Gradio interface
# demo = gr.Interface(
#     fn=bn_postagger,
#     inputs=[
#         gr.Textbox(label="Enter Bangla Sentence", placeholder="বাংলা বাক্য লিখুন"), 
#         gr.Dropdown(["Google", "BanglaNMT", "MyMemory"], label="Select a Translator"),
#         gr.Dropdown(["spaCy", "NLTK", "Flair", "TextBlob"], label="Select a PoS Tagger")
#     ],
#     outputs= [
#         gr.Textbox(label="English Translation"), 
#         RichTextbox(label="PoS Tags"),
#         gr.Textbox(label="Overall PoS Tagging Accuracy")
#     ],
#     live=False,
#     title="Bangla PoS Taggers",
#     theme='',
#     examples=[
#         ["বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র।"],
#         ["বাংলাদেশের সংবিধানিক নাম কি?"],
#         ["বাংলাদেশের সাংবিধানিক নাম গণপ্রজাতন্ত্রী বাংলাদেশ।"],
#         ["তিনজনের কেউই বাবার পথ ধরে প্রযুক্তি দুনিয়ায় হাঁটেননি।"],
#         ["বিশ্বের আরও একটি সেরা ক্লাব।"]

#     ]
# )

with gr.Blocks(css="styles.css") as demo:
    gr.HTML("<h1>Bangla PoS Taggers</h1>")
    gr.HTML("<p>Parts of Speech (PoS) Tagging of Bangla Sentence using Bangla-English <strong>Word Alignment</strong></p>")

    with gr.Row():
        with gr.Column():
            inputs = [
                gr.Textbox(
                    label="Enter Bangla Sentence", 
                    placeholder="বাংলা বাক্য লিখুন"
                ),
                gr.Dropdown(
                    choices=["Google", "BanglaNMT", "MyMemory"], 
                    label="Select a Translator"
                ),
                gr.Dropdown(
                    choices=["spaCy", "NLTK", "Flair", "TextBlob"], 
                    label="Select a PoS Tagger"
                )
            ]

            btn = gr.Button(value="Submit", elem_classes="mybtn")
            gr.ClearButton(inputs)

        with gr.Column():
            outputs = [
                gr.Textbox(label="English Translation"), 
                RichTextbox(label="PoS Tags"),
                gr.Textbox(label="Overall PoS Tagging Accuracy")
            ]

    btn.click(bn_postagger, inputs, outputs)

    gr.Examples([
        ["বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র।", "Google", "NLTK"],
        ["বাংলাদেশের সংবিধানিক নাম কি?", "Google", "spaCy"],
        ["বাংলাদেশের সাংবিধানিক নাম গণপ্রজাতন্ত্রী বাংলাদেশ।", "Google", "TextBlob"],
        ["তিনজনের কেউই বাবার পথ ধরে প্রযুক্তি দুনিয়ায় হাঁটেননি।", "Google", "spaCy"],
        ["তিনজনের কেউই বাবার পথ ধরে প্রযুক্তি দুনিয়ায় হাঁটেননি।", "BanglaNMT", "spaCy"],
        ["তিনজনের কেউই বাবার পথ ধরে প্রযুক্তি দুনিয়ায় হাঁটেননি।", "MyMemory", "spaCy"],
        ["বিশ্বের আরও একটি সেরা ক্লাব।", "Google", "Flair"]

    ], inputs)



# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()