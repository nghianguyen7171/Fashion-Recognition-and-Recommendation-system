import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
import time
from plotly.offline import iplot

import itertools
import os
import warnings
warnings.filterwarnings('ignore')
plt.rc('font',family='Times New Roman')

import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn import metrics
from model import CompatModel

import resnet
from utils import prepare_dataloaders
# warnings.filterwarnings('ignore')
# plt.rc('font',family='Times New Roman')
plt.switch_backend('agg')

def app():
    header = st.container()
    recom = st.container()

    with header:
        st.write('Recommendation for a set of fashion items')
        #main_logo_path = 'logo/logo_cnu.png'
        #main_logo = Image.open(main_logo_path).resize((200, 200))

    with recom:
        # Dataloader
        train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = prepare_dataloaders()

        iterator = iter(test_loader)
        batch = next(iterator)
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        # Model
        device = torch.device("cuda:0")
        model = CompatModel(embed_size=1000, need_rep=True, vocabulary=len(train_dataset.vocabulary)).to(device)

        # Load model weights
        model.load_state_dict(torch.load('./model_train.pth'))
        model.eval()

        # Function
        def loadimg_from_id(ID, root_dir=test_dataset.root_dir):
            """load image from pre-defined id.
            
            Args:
                ID: List of ids of 5 items.
                
            Return:
                imgs: torch.tensor of shape (1, 5, 3, 224, 224)
            """
            imgs = []
            for id in ID:
                if 'mean' in id:
                    img_path = os.path.join(test_dataset.data_dir, id.split('_')[0]) + '.png'
                else:
                    img_path = os.path.join(root_dir, *id.split('_')) + '.jpg'
                img = Image.open(img_path).convert('RGB')
                img = test_dataset.transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs)
            imgs = imgs.unsqueeze(0)
            return imgs

        def defect_detect(img, model, normalize=True):
            """ Compute the gradients of each element in the comparison matrices to 
            approximate the problem of each input.
            
            Args:
                img: images of shape (N, 3, 224, 224).
                model: the model to compute the compatibility score.
                normalize: whether to normalize the relation results.
                
            Return:
                relation: gradients on comparison matrix.
                out: prediction score.
            """
            # Register hook for comparison matrix
            relation = None
            def func_r(module, grad_in, grad_out):
                nonlocal relation
                relation = grad_in[1].detach()

            for name, module in model.named_modules():
                if name == 'predictor.0':
                    module.register_backward_hook(func_r)

            # Forward
            out, *_ = model._compute_score(img)
            one_hot = torch.FloatTensor([[-1]]).to(device)

            # Backward
            model.zero_grad()
            out.backward(gradient=one_hot, retain_graph=True)
            
            if normalize:
                relation = relation / (relation.max() - relation.min())
            relation += 1e-3
            return relation, out.item()

        def vec2mat(relation, select):
            """ Convert relation vector to 4 matrix, which is corresponding to 4 layers
            in the backend CNN.
            
            Args:
                relation: (np.array | torch.tensor) of shpae (60,)
                select: List of select item indices, e.g. (0, 2, 3) means select 3 items
                    in total 5 items in the outfit.
                
            Return:
                mats: List of matrix
            """
            mats = []
            for idx in range(4):
                mat = torch.zeros(5, 5)
                mat[np.triu_indices(5)] = relation[15*idx:15*(idx+1)]
                mat += torch.triu(mat, 1).transpose(0, 1)
                mat = mat[select, :]
                mat = mat[:, select]
                mats.append(mat)
            return mats

        def show_rela_diagnosis(relation, select, cmap=plt.cm.Blues):
            """ Visualize diagnosis on relationships of 4 scales
            
            Args:
                relation: (np.array | torch.tensor) of shpae (60,)
                select: List of select item indices
            """
            mats = vec2mat(relation , select)
                
            fig = plt.figure(figsize=(20, 10))
            all_names = {0:'Top', 1:'Bottom', 2:'Shoe', 3:'Bag', 4:'Accssory'}
            node_names = {i:all_names[s] for i, s in enumerate(select)}
            
            edge_vmax = max(m.max() for m in mats)
            edge_vmin = min(m.min() for m in mats)
            
            container = []
            for idx in range(4):
                A = mats[idx]
                if isinstance(A, torch.Tensor):
                    A = A.cpu().data.numpy()
                        
                A = np.triu(A, k=1)
                A = np.round(A, decimals=2)
                container.append(A)
            container = np.stack(container)
            sorted_vedge = sorted(container.ravel(), reverse=True)
                
            for idx in range(4):
                plt.subplot(1, 4, idx+1)
                plt.title("Layer {}".format(idx+1), fontsize=28)
                A = mats[idx]
                if isinstance(A, torch.Tensor):
                    A = A.cpu().data.numpy()
                        
                A = np.triu(A, k=1)
                A = np.round(A, decimals=2)
                indices = np.triu_indices(A.shape[0], k=1)
                weights = A[indices[0], indices[1]]
                # Generate graph
                G = nx.Graph()
                for i, j, weight in zip(*indices, weights):
                    G.add_edge(node_names[i], node_names[j], weight=weight)
                
                elarge, esmall, filtered_weights = [], [], []
                for e in G.edges(data=True):
                    if e[2]['weight'] in sorted_vedge[:3]:
                        elarge.append((e[0], e[1]))
                    else:
                        esmall.append((e[0], e[1]))
                        filtered_weights.append(e[2]['weight'])
                pos=nx.circular_layout(G) # positions for all nodes

                # nodes
                nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes()], node_size=1600, node_color='#A0CBE2')

                # edges
                nx.draw_networkx_edges(G,pos,edgelist=esmall, width=6, alpha=0.5, edge_color=filtered_weights, edge_cmap=cmap,
                                    edge_vmax=edge_vmax, edge_vmin=edge_vmin)
                nx.draw_networkx_edges(G,pos,edgelist=elarge, width=6, alpha=0.5, edge_color='red', style='dashed')

                # labels
                labels = nx.get_edge_attributes(G,'weight')
                nx.draw_networkx_labels(G,pos, font_size=18, font_family='Times New Roman')
                if len(select) == 4:
                    nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels, label_pos=0.33)
                else:
                    nx.draw_networkx_edge_labels(G, pos, font_size=18, font_family='Times New Roman', edge_labels=labels)
                
                plt.axis('off')
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
            plt.tight_layout()
            # plt.show()
            st.pyplot(fig)

        def item_diagnosis(relation, select):
            """ Output the most incompatible item in the outfit
            
            Return:
                result (list): Diagnosis value of each item 
                order (list): The indices of items ordered by its importance
            """
            mats = vec2mat(relation, select)
            for m in mats:
                mask = torch.eye(*m.shape).byte()
                m.masked_fill_(mask, 0)
            result = torch.cat(mats).sum(dim=0)
            order = [i for i, j in sorted(enumerate(result), key=lambda x:x[1], reverse=True)]
            return result, order

        def show_imgs(x, select=None):
            """ Show multiple items in a outfit.
            
            Args:
                x: torch.tensor of shape(5, 3, 224, 224)
                select: List of selected item index
            """
            if select is None:
                select = list(range(5))
            fig = plt.figure(figsize=(5*len(select), 5))
            for i, s in enumerate(select):
                plt.subplot(1, len(select), i+1)
                img = x[s]
                img = img.cpu().data.numpy().transpose((1, 2, 0)) * 255
                img = img[..., :3]
                plt.gca().axis('off')
                plt.imshow(np.uint8(img))
            st.pyplot(fig)


        def retrieve_sub(x, select, order):
            """ Retrieve the datset to substitute the worst item for the best choice.
            """
            all_names = {0:'upper', 1:'bottom', 2:'shoe', 3:'bag', 4:'accessory'}
        
            best_score = -1
            best_img_path = dict()

            for o in order:
                if best_score > 0.9:
                    break
                problem_part_idx = select[o]
                problem_part = all_names[problem_part_idx]
                for outfit in test_dataset.data:
                    if best_score > 0.9:
                        break
                    if problem_part in outfit[1]:
                        img_path = os.path.join(test_dataset.root_dir, outfit[0], str(outfit[1][problem_part]['index'])) + '.jpg'
                        img = Image.open(img_path).convert('RGB')
                        img = test_dataset.transform(img).to(device)
                        x[0][problem_part_idx] = img
                        with torch.no_grad():
                            score, *_ = model._compute_score(x)
                        if score.item() > best_score:
                            best_score = score.item()
                            best_img_path[problem_part] = img_path
                x[0][problem_part_idx] = test_dataset.transform(Image.open(best_img_path[problem_part]).convert('RGB')).to(device)
            
                st.write('problem_part: {}'.format(problem_part))
                st.write('best substitution: {}'.format(best_img_path[problem_part]))
                st.write('After substitution the score is {:.4f}'.format(best_score))

                fig = plt.figure(figsize=(5, 5))
                plt.imshow(plt.imread(best_img_path[problem_part]))
                plt.gca().axis('off')
                plt.title('Best substitution for {}'.format(problem_part))
                
                st.pyplot(fig)
            st.write('Fashion sets after recommendation')
            show_imgs(x[0], select)
            return best_score, best_img_path

        Input_id1 = st.text_input('Input the top id', '178118160_1')
        Input_id2 = st.text_input('Input the bottom id', 'bottom_mean')
        Input_id3 = st.text_input('Input the shoes id', '199285568_4')
        Input_id4 = st.text_input('Input the bag id', '111355382_5')
        Input_id5 = st.text_input('Input the accessory id', '209432387_4')

        # if input_id1 and input_id2 and input_id3 and input_id4 and input_id5 not not None:
        if Input_id1 and Input_id2 and Input_id3 and Input_id4 and Input_id5:
            ID = [Input_id1, Input_id2, Input_id3, Input_id4, Input_id5]
            x = loadimg_from_id(ID).to(device)
            # kick out the mean images for padding the sequence when making visualization
            select = [i for i, l in enumerate(ID) if 'mean' not in l]
            if st.button('Show input fashion set'):
                # Step 1: show images in an outfit
                st.write('The input outfits are:')
                show_imgs(x[0], select)
            
            if st.button('Show dignosis results'):
                # Step 2: show diagnosis results
                relation, out = defect_detect(x, model)
                relation = relation.squeeze().cpu().data
                st.write('The diagnosis results are:')
                show_rela_diagnosis(relation, select, cmap=plt.cm.Blues)
                result, order = item_diagnosis(relation, select)
                st.write("Predicted Score: {:.4f}\nProblem value of each item: {}\nOrder: {}\n".format(out, result, order))
                #st.write('The most incompatible item is: {}'.format(ID[order[0]]))
                st.write("="*88)

            #if st.button('show recommendation results'):
                st.write('Substitution results')
                # Step 3: substitute the problem items for revision
                best_score, best_img_path = retrieve_sub(x, select, order)
                st.write("="*88)


        
        
                        

