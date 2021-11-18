"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import pickle
from azureml.core import Workspace
from azureml.core.run import Run
import os
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import json
import subprocess
from typing import Tuple, List
from azureml.core import Run, Dataset

# run_history_name = 'devops-ai'
# os.makedirs('./outputs', exist_ok=True)
# #ws.get_details()
# Start recording results to AML
# run = Run.start_logging(workspace = ws, history_name = run_history_name)

# If you want to pass dataset as parameter uncomment below 3 lines

run = Run.get_context()
train_ds = run.input_datasets['train_ds']
df = train_ds.to_pandas_dataframe()


parser = argparse.ArgumentParser()
parser.add_argument("--synapse_runid", type=str, help="synapse_runid")
parser.add_argument("--input", type=str, help="input")
parser.add_argument("--output", type=str, help="output")


args = parser.parse_args()
print(f"Print run ID: {args.synapse_runid}")

x_columns = ["age", "gender", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
y_columns = ["Y"]
df.columns = x_columns + y_columns

X_train, X_test, y_train, y_test = train_test_split(df[x_columns], df[y_columns], test_size=0.2, random_state=0)
data = {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test, "y": y_test}}

print("Running train.py")

# Randomly pic alpha
alphas = np.arange(0.0, 1.0, 0.05)
alpha = alphas[np.random.choice(alphas.shape[0], 1, replace=False)][0]
print(alpha)
run.log("alpha", alpha)
reg = Ridge(alpha=alpha)
reg.fit(data["train"]["X"], data["train"]["y"])
preds = reg.predict(data["test"]["X"])
run.log("mse", mean_squared_error(preds, data["test"]["y"]))


# Save model as part of the run history
model_path = "outputs/sklearn_regression_model.pkl"
# model_name = "."

os.makedirs('outputs', exist_ok=True)

joblib.dump(value=reg, filename=model_path)
run.parent.upload_file(model_path, model_path)
