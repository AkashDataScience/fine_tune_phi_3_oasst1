import pandas as pd
from tqdm import tqdm
from treelib import Tree
from datasets import load_dataset, Dataset

def add_tree_level(df):
    """helper function to add tree level to a df"""

    # if tree level already exists, return df
    if "tree_level" in df.columns:
        return df

    else:
        tree_level_map = {}

        # iterate over rows in df
        for i, row in df.iterrows():
            message_id = row["message_id"]
            parent_id = row["parent_id"]

            # if parent_id is None, then it is a root message
            if parent_id is None:
                tree_level_map[message_id] = 0
            # if parent_id is the same as message_tree_id, then it is a direct reply to the root message
            elif parent_id == row["message_tree_id"]:
                tree_level_map[message_id] = 1
            # else just look up the tree level of the parent_id and add 1
            else:
                tree_level_map[message_id] = tree_level_map[parent_id] + 1

        # create a df from the tree_level_map and merge it with the original df
        df_tree_level_map = (
            pd.DataFrame.from_dict(tree_level_map, orient="index", columns=["tree_level"])
            .reset_index()
            .rename(columns={"index": "message_id"})
        )

        return df.merge(df_tree_level_map, on="message_id")
    
def list_to_dict(chat_list):
  chat_dict = []
  for chat in chat_list:
    for chat_index in range(0, len(chat), 2):
      if chat_index+1 < len(chat):
        chat_sample = chat[:chat_index+2]
        role = "user"
        chat_sample_dict = []
        for message in chat_sample:
          message_role = {"content":message, "role":role}
          chat_sample_dict.append(message_role)
          role = "assistant" if role == "user" else "user"
        chat_dict.append(chat_sample_dict)
  return chat_dict

def get_chat_list(df_tree):
  # lets create a tree of message texts
  text_tree = Tree()

  # iterate over rows in df_message_tree
  for i, row in df_tree.iterrows():
      # grab the message_id, parent_id, text, and parent text
      message_id = row["message_id"]
      parent_id = row["parent_id"]
      text = row["text"]
      #text_tag = text.replace("\n", " ")
      parent_text = (
          df_tree.query(f"message_id == '{parent_id}'")["text"].values[0] if parent_id is not None else "ROOT"
      )

      # if parent_id is None, then it is a root message so dont add parent text as is none
      if parent_id is None:
          text_tree.create_node(text, text)
      # else use the parent text short as the parent
      elif not text_tree.contains(text):
          text_tree.create_node(text, text, parent=parent_text)

  chat_list = text_tree.paths_to_leaves()

  chat_dict = list_to_dict(chat_list)

  return chat_dict

def df_to_chat_list(df):
  message_tree_id_list = list(df[df["parent_id"].isin([None])]['message_id'])
  chat_dict_list = []
  for message_tree_id in tqdm(message_tree_id_list):
    # look at all data for this message tree
    df_message_tree = df.query(f"message_tree_id == '{message_tree_id}'").sort_values("created_date")

    # add tree level to df
    df_message_tree = add_tree_level(df_message_tree)

    chat_dict_list.extend(get_chat_list(df_message_tree))

  return chat_dict_list

def get_train_test_data():
    dataset_name = "OpenAssistant/oasst1"

    dataset = load_dataset(dataset_name)
    train = dataset['train']
    val = dataset['validation']

    train_df = train.to_pandas()
    val_df = val.to_pandas()

    train_chat_list = df_to_chat_list(train_df)
    val_chat_list = df_to_chat_list(val_df)

    train_ds = Dataset.from_dict({"messages": train_chat_list})
    val_ds = Dataset.from_dict({"messages": val_chat_list})

    column_names = list(train_ds.features)

    return train_ds, val_ds, column_names