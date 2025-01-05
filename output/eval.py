def f1(tp, tn, fp, fn):
    return {
        "Precision": tp/(tp+fp),
        "Recall": tp/(tp+fn),
        "F1-score": 2*tp/(2*tp+fp+fn),
        "Accuracy": (tp + tn)/(tp + tn + fp + fn)
    }

true_wm = []
pred_wm = []

with open ("pred_wm.txt") as f:
    for wm in f:
        pred_wm.append(wm)
with open ("true_wm.txt") as f:
    for wm in f:
        true_wm.append(wm)
        
        
tp = 1e-7
fp = 1e-7
tn = 1e-7
fn = 1e-7
# Bit accuracy of decoding only images that are predicted to be watermarked
avg_pred_bit_acc = 0
# Bit accuracy of decoding all images that are watermarked
avg_total_bit_acc = 0

for i in range(len(true_wm)):
    true_label = (true_wm[i] != 2)
    pred_label = (pred_wm[i] != 2)
    if pred_label == True and true_label == pred_label:
        tp += 1
    if pred_label == False and true_label == pred_label:
        tn += 1 
    if pred_label == True and true_label != pred_label:
        fp += 1 
    if pred_label == False and true_label != pred_label:
        fn += 1 
    if pred_label == 1:
        num_correct = 0
        for j in range(len(true_wm[i])):
            if true_wm[i][j] == pred_wm[i][j]:
                num_correct += 1
        num_correct /= len(true_wm[i])
        avg_pred_bit_acc += num_correct
    if true_label == 1:
        num_correct = 0
        for j in range(len(true_wm[i])):
            if true_wm[i][j] == pred_wm[i][j]:
                num_correct += 1
        num_correct /= len(true_wm[i])
        avg_total_bit_acc += num_correct
        

avg_pred_bit_acc /= len(true_wm)
avg_total_bit_acc /= len(true_wm)

print(f1(tp, tn, fp, tn))
print(f"Avg bit acc on images predicted as watermarked: {avg_pred_bit_acc}")
print(f"Avg bit acc on all watermarked images: {avg_total_bit_acc}")
        