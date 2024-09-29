import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

data = np.load("normalized_data2.npy", allow_pickle=True)

fig, ax = plt.subplots()

vector_good = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='green')
vector_bad = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='red')
vector_love = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='pink')
vector_hate = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='firebrick')
vector_happy = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='gold')
vector_sad = ax.quiver(0,0,0,0,angles='xy',scale_units='xy',scale=1, color='dodgerblue')

label_good = ax.text(0, 0, 'Good', color='#2E7D32')  # Dark Green
label_bad = ax.text(0, 0, 'Bad', color='#C62828')    # Dark Red
label_love = ax.text(0, 0, 'Love', color='#D81B60')  # Dark Pink
label_hate = ax.text(0, 0, 'Hate', color='#7B1FA2')  # Dark Purple
label_happy = ax.text(0, 0, 'Happy', color='#B8860B') 
label_sad = ax.text(0, 0, 'Sad', color='royalblue')  # Dark Blue
label_iter = ax.text(0, 0, 'Iteration', color='black')  # Black for iteration
label_iter.set_position((0, -1.4))
label_iter.set_horizontalalignment('center')

circle = patches.Circle((0, 0), 1,edgecolor='k' ,fill=False, zorder=3)
ax.add_patch(circle)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_title('PCA of Word Embeddings of BERT')
ax.axhline(0, color='k')
ax.axvline(0, color='k')
ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect('equal')

start_time = time.time()
for i in range(len(data)):
    vector_good.set_UVC(data[i:i+1,0,0], data[i:i+1,0,1])
    vector_bad.set_UVC(data[i:i+1,1,0], data[i:i+1,1,1])
    vector_love.set_UVC(data[i:i+1,2,0], data[i:i+1,2,1])
    vector_hate.set_UVC(data[i:i+1,3,0], data[i:i+1,3,1])
    vector_happy.set_UVC(data[i:i+1,4,0], data[i:i+1,4,1])
    vector_sad.set_UVC(data[i:i+1,5,0], data[i:i+1,5,1])

    label_good.set_position((data[i:i+1,0,0], data[i:i+1,0,1]))
    label_good.set_horizontalalignment('left' if data[i:i+1,0,0] > 0 else 'right')
    label_good.set_verticalalignment('bottom' if data[i:i+1,0,1] > 0 else 'top')

    label_bad.set_position((data[i:i+1,1,0], data[i:i+1,1,1]))
    label_bad.set_horizontalalignment('left' if data[i:i+1,1,0] > 0 else 'right')
    label_bad.set_verticalalignment('bottom' if data[i:i+1,1,1] > 0 else 'top')

    label_love.set_position((data[i:i+1,2,0], data[i:i+1,2,1]))
    label_love.set_horizontalalignment('left' if data[i:i+1,2,0] > 0 else 'right')
    label_love.set_verticalalignment('bottom' if data[i:i+1,2,1] > 0 else 'top')

    label_hate.set_position((data[i:i+1,3,0], data[i:i+1,3,1]))
    label_hate.set_horizontalalignment('left' if data[i:i+1,3,0] > 0 else 'right')
    label_hate.set_verticalalignment('bottom' if data[i:i+1,3,1] > 0 else 'top')

    label_happy.set_position((data[i:i+1,4,0], data[i:i+1,4,1]))
    label_happy.set_horizontalalignment('left' if data[i:i+1,4,0] > 0 else 'right')
    label_happy.set_verticalalignment('bottom' if data[i:i+1,4,1] > 0 else 'top')

    label_sad.set_position((data[i:i+1,5,0], data[i:i+1,5,1]))
    label_sad.set_horizontalalignment('left' if data[i:i+1,5,0] > 0 else 'right')
    label_sad.set_verticalalignment('bottom' if data[i:i+1,5,1] > 0 else 'top')

    label_iter.set_text("Iteration: "+str(i))

    plt.draw()
    plt.pause(0.001)

print("Time taken: ",time.time() - start_time)
plt.show()

