# # # classifier.py

# # from sklearn.datasets import load_wine
# # from sklearn.model_selection import train_test_split
# # import pandas as pd
# # from sklearn.metrics import accuracy_score
# # from RandomForest import RandomForest

# # # Load dataset
# # data = load_wine()
# # X = pd.DataFrame(data.data, columns=data.feature_names)
# # y = pd.Series(data.target)

# # # Split dataset
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # # Instantiate RandomForest with adjusted parameters
# # rf = RandomForest(n_trees=30, max_depth=10, min_samples_split=5, min_samples_leaf=7, 
# #                   n_feature="sqrt", bootstrap=True, oob=True, criterion="gini", 
# #                   treetype="classification", random_state=42)

# # # Fit the model
# # rf.fit(X_train, y_train)

# # # Check the structure and depth of the first tree before pruning
# # print("Tree structure before pruning:")
# # print(rf.trees[0].node_id_dict)
# # print(f"Tree depth before pruning: {rf.trees[0].max_depth_}")

# # # Prune the forest
# # rf.prune(min_samples_leaf=15)

# # # Check the structure and depth of the first tree after pruning
# # print("Tree structure after pruning:")
# # print(rf.trees[0].node_id_dict)
# # print(f"Tree depth after pruning: {rf.trees[0].max_depth_}")

# # # Print detailed information about pruned nodes
# # for i, tree in enumerate(rf.trees):
# #     print(f"\nTree {i}:")
# #     for node_id, node_info in tree.node_id_dict.items():
# #         print(f"Node ID: {node_id}, Depth: {node_info['depth']}, Is Leaf: {node_info['is_leaf_node']}, Samples: {node_info['samples']}")

# # # Evaluate the model
# # y_pred = rf.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f'Accuracy: {accuracy:.2f}')
# # print(f'OOB Score: {rf.oob_score:.2f}')


# # # """just tell me is it worked ? If so, then, I would like to request you to draw the tree after and before pruning so that I can visualise"""











# # # decision_tree_pruning.py

# # from sklearn.datasets import make_classification
# # from sklearn.tree import DecisionTreeClassifier, plot_tree
# # import matplotlib.pyplot as plt

# # # Generate synthetic dataset
# # X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# # # Fit a decision tree classifier
# # clf = DecisionTreeClassifier(random_state=42)
# # clf.fit(X, y)

# # # Plot the decision tree before pruning
# # plt.figure(figsize=(20, 10))
# # plot_tree(clf, filled=True)
# # plt.title("Decision Tree Before Pruning")
# # plt.savefig("decision_tree_before_pruning.png")

# # # Prune the decision tree by setting max depth
# # clf_pruned = DecisionTreeClassifier(random_state=42, max_depth=3)
# # clf_pruned.fit(X, y)

# # # Plot the decision tree after pruning
# # plt.figure(figsize=(20, 10))
# # plot_tree(clf_pruned, filled=True)
# # plt.title("Decision Tree After Pruning")
# # plt.savefig("decision_tree_after_pruning.png")



# # classifier.py

# # classifier.py

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_wine
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from RandomForest import RandomForest
# from graphviz import Digraph

# def visualize_tree(tree, feature_names, title):
#     """ Visualizes a single decision tree. """
#     dot = Digraph()

#     def add_nodes_edges(dot, tree, node_id, depth):
#         if node_id not in tree:
#             return
        
#         # Add node
#         node = tree[node_id]
#         if node['is_leaf_node']:
#             label = f"Leaf\nsamples={node['samples']}\nvalue={node['value']}\n"
#             label += f"dist={node['value_distribution']}"
#             dot.node(name=str(node_id), label=label, shape='box')
#         else:
#             label = f"{node['feature']}\n<threshold: {node['threshold']}>\ngini={node['gini']:.4f}\nsamples={node['samples']}"
#             dot.node(name=str(node_id), label=label)
        
#         # Add edges
#         if not node['is_leaf_node']:
#             add_nodes_edges(dot, tree, 2 * node_id + 1, depth + 1)
#             add_nodes_edges(dot, tree, 2 * node_id + 2, depth + 1)
#             dot.edge(str(node_id), str(2 * node_id + 1), "True")
#             dot.edge(str(node_id), str(2 * node_id + 2), "False")

#     add_nodes_edges(dot, tree, 0, 0)
#     dot.render(title, format='png', cleanup=True)

# # Load dataset
# data = load_wine()
# X = pd.DataFrame(data.data, columns=data.feature_names)
# y = pd.Series(data.target)

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Instantiate RandomForest with adjusted parameters
# rf = RandomForest(n_trees=1, max_depth=10, min_samples_split=5, min_samples_leaf=7, 
#                   n_feature="sqrt", bootstrap=True, oob=True, criterion="gini", 
#                   treetype="classification", random_state=42)

# # Fit the model
# rf.fit(X_train, y_train)

# # Check the structure and depth of the first tree before pruning
# print("Tree structure before pruning:")
# print(rf.trees[0].node_id_dict)
# print(f"Tree depth before pruning: {rf.trees[0].max_depth_}")

# # Visualize the tree before pruning
# visualize_tree(rf.trees[0].node_id_dict, X.columns, "decision_tree_before_pruning")

# # Prune the forest
# rf.prune(min_samples_leaf=15)

# # Check the structure and depth of the first tree after pruning
# print("Tree structure after pruning:")
# print(rf.trees[0].node_id_dict)
# print(f"Tree depth after pruning: {rf.trees[0].max_depth_}")

# # Visualize the tree after pruning
# visualize_tree(rf.trees[0].node_id_dict, X.columns, "decision_tree_after_pruning")

# # Evaluate the model
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
# print(f'OOB Score: {rf.oob_score:.2f}')

# # Display the images
# img_before = plt.imread("decision_tree_before_pruning.png")
# img_after = plt.imread("decision_tree_after_pruning.png")

# fig, axs = plt.subplots(1, 2, figsize=(20, 10))
# axs[0].imshow(img_before)
# axs[0].set_title("Decision Tree Before Pruning")
# axs[0].axis('off')

# axs[1].imshow(img_after)
# axs[1].set_title("Decision Tree After Pruning")
# axs[1].axis('off')

# plt.show()


# classifier.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from graphviz import Digraph

def visualize_tree(tree, feature_names, title):
    """ Visualizes a single decision tree. """
    dot = Digraph()

    def add_nodes_edges(dot, tree, node_id, depth):
        if node_id not in tree:
            return
        
        # Add node
        node = tree[node_id]
        if node['is_leaf_node']:
            label = f"Leaf\nsamples={node['samples']}\nvalue={node['value']}\n"
            label += f"dist={node['value_distribution']}"
            dot.node(name=str(node_id), label=label, shape='box')
        else:
            label = f"{node['feature']}\n<threshold: {node['threshold']}>\ngini={node['gini']:.4f}\nsamples={node['samples']}"
            dot.node(name=str(node_id), label=label)
        
        # Add edges
        if not node['is_leaf_node']:
            add_nodes_edges(dot, tree, 2 * node_id + 1, depth + 1)
            add_nodes_edges(dot, tree, 2 * node_id + 2, depth + 1)
            dot.edge(str(node_id), str(2 * node_id + 1), "True")
            dot.edge(str(node_id), str(2 * node_id + 2), "False")

    add_nodes_edges(dot, tree, 0, 0)
    dot.render(title, format='png', cleanup=True)

# Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate RandomForest with adjusted parameters
rf = RandomForest(n_trees=1, max_depth=10, min_samples_split=5, min_samples_leaf=7, 
                  n_feature="sqrt", bootstrap=True, oob=True, criterion="gini", 
                  treetype="classification", random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Check the structure and depth of the first tree before pruning
print("Tree structure before pruning:")
print(rf.trees[0].node_id_dict)
print(f"Tree depth before pruning: {rf.trees[0].max_depth_}")

# Visualize the tree before pruning
visualize_tree(rf.trees[0].node_id_dict, X.columns, "decision_tree_before_pruning")

# Prune the forest
rf.prune(min_samples_leaf=15)

# Check the structure and depth of the first tree after pruning
print("Tree structure after pruning:")
print(rf.trees[0].node_id_dict)
print(f"Tree depth after pruning: {rf.trees[0].max_depth_}")

# Visualize the tree after pruning
visualize_tree(rf.trees[0].node_id_dict, X.columns, "decision_tree_after_pruning")

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'OOB Score: {rf.oob_score:.2f}')

# Display the images
img_before = plt.imread("decision_tree_before_pruning.png")
img_after = plt.imread("decision_tree_after_pruning.png")

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(img_before)
axs[0].set_title("Decision Tree Before Pruning")
axs[0].axis('off')

axs[1].imshow(img_after)
axs[1].set_title("Decision Tree After Pruning")
axs[1].axis('off')

plt.show()
