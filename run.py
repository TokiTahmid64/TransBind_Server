import streamlit as st
import re
import torch
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# local_directory = "."

# # Load the tokenizer and model from the local directory
# tokenizer = T5Tokenizer.from_pretrained(local_directory, do_lower_case=False)
# model = T5EncoderModel.from_pretrained(local_directory)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm1d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_1x1,
        out_3x3,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1) # 
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_1x1, kernel_size=1, padding=0),
            ConvBlock(red_1x1, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_1x1, kernel_size=1),
            ConvBlock(red_1x1, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        #print((self.branch1(x)).shape)
        return (torch.cat([branch(x) for branch in branches], 1))

class Baseline_1(nn.Module):
    def __init__(
        self, 

       
    ):
        super(Baseline_1, self).__init__()
        self.inception_1=InceptionBlock(1024,128,128,128,128,128)

        self.inception_2=InceptionBlock(512,64,64,64,64,64)

        self.inception_3=InceptionBlock(256,32,32,32,32,32)
        self.linear_1=nn.Linear(128,32)
        self.linear_2=nn.Linear(32,2)
        self.dropout = nn.Dropout(p=0.2)






    def forward(self, x):
        x=torch.transpose(x,2,1)
        x=self.inception_1(x)
        x=self.dropout(x)
        x=self.inception_2(x)
        x=self.dropout(x)
        x=self.inception_3(x)
        x=self.dropout(x)
        x=torch.mean(x,dim=2)
        #x = x.reshape(x.shape[0], -1)
        x=F.relu(self.linear_1(x))
        x= self.linear_2(x)
        x = F.softmax(x, dim=1)
        
        return x
    
    

model_predictor = Baseline_1()
device = torch.device('cpu')
model_predictor.to(device)

model_predictor.load_state_dict(torch.load('model.pt'))

print("Model loaded")




# Function to generate embeddings
def generate_embedding(seq):
    model.eval()  # Set the model to evaluation mode

    # Tokenize the protein sequence
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Return embeddings
    return outputs.last_hidden_state[:, 1:, :].cpu().numpy()





def get_prediction(embedding):
    model_predictor.eval()
    embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
    # embedding = torch.transpose(embedding, 2, 1)
    with torch.no_grad():
        outputs = model_predictor(embedding)
    print(outputs.shape)
    return outputs


def draw_protein_sequence(seq, results):
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.axis('off')
    
    x = np.linspace(0, len(seq), len(seq))
    y = np.sin(x / 3)  # Helical effect
    
    colors = ['red' if result.item() == 1 else 'blue' for result in results]
    
    ax.scatter(x, y, c=colors, s=100, edgecolors='w', zorder=2)
    
    for i, amino_acid in enumerate(seq):
        ax.text(x[i], y[i] + 0.15, amino_acid, ha='center', va='center', fontsize=12, zorder=3)
    
    ax.plot(x, y, color='gray', zorder=1)
    
    return fig

def display_results(seq, results):
    amino_acid_full_names = {
        'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic acid',
        'C': 'Cysteine', 'E': 'Glutamic acid', 'Q': 'Glutamine', 'G': 'Glycine',
        'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
        'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
        'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
    }
    
    results_text = ["Binds to Nucleic acid" if result.item() == 1 else "Does not Bind to Nucleic acid" for result in results]
    
    data = {
        "Index": list(range(1, len(seq.split()) + 1)),
        "Amino Acid": [f"{aa} ({amino_acid_full_names[aa]})" for aa in seq.split()],
        "Result": [result.item() for result in results],
        "Comment": results_text
    }
    
    st.dataframe(data, height=600)


# Main Streamlit app function
def main():
    # Create columns for the title and the image
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<h2 style="color: green;">TransBind: Sequence-Only Precise Detection of DNA-binding Proteins and Residues Using Language Models and Deep Learning</h2>', unsafe_allow_html=True)

    with col2:
        # Display an image from a local file
        local_image_path = "image.jpg"
        st.image(local_image_path, width=300)

    # Text input for protein sequence
    seq = st.text_area('Enter the protein sequence:')

    # Submit button
    if st.button('Submit'):
        if seq:
            start_time = time.time()  # Start the timer
            
            with st.spinner('Processing...'):
                # Clean and process sequence
                seq = re.findall(r'[A-Z]', seq)
                seq = ' '.join(seq)

                if len(seq) > 1024:
                    st.warning("Sequence length is greater than 1024. Truncating the sequence.")
                    seq = seq[:1024]

                seq = ' '.join(seq)

                # Generate embeddings
                embedding = generate_embedding(seq)
                embedding = embedding.squeeze(0)

                results = []

                # Get prediction
                for embd in embedding:
                    embd = embd.reshape(1, 1, embd.shape[0])
                    prediction = get_prediction(embd)
                    prediction = torch.argmax(prediction, dim=1)
                    results.append(prediction)

                # Display the results in a scrollable table
                display_results(seq, results)
                
                # Draw the protein sequence
                fig = draw_protein_sequence(seq.split(), results)
                st.pyplot(fig)
            
            end_time = time.time()  # End the timer
            duration = end_time - start_time  # Calculate the duration
            st.success(f"Process completed in {duration:.2f} seconds.")
            
        else:
            st.warning("Please enter a protein sequence.")

    # Refresh button
    if st.button('Refresh'):
        # Clear the text area
        seq = ''

if __name__ == '__main__':
    main()