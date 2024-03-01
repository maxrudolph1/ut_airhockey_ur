import pickle
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

data = pickle.load(open('../data/measured_values.pkl', 'rb'))


servo_poses = np.array([d[0][0][:2] for d in data])
poses = np.array([d[1] for d in data])
true_speed = np.array([d[2] for d in data])
true_force = np.array([d[3] for d in data])
acc = np.array([d[4] for d in data])


print("speed shape", true_speed.shape)
print("foce shape", true_force.shape)
print("acc shape", acc.shape)

idx_zero_speed = np.where(true_speed.sum(axis=-1) == 0)[0]
t = np.arange(acc.shape[0])
plt.figure()
plt.title('Speed in all 6 DOF')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.plot(t, true_speed[:,i])

plt.figure()
plt.title('Force in all 6 DOF')
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.plot(t, true_force[:,i])

plt.figure()
plt.title('acc in all 6 DOF')
for i in range(3):
    plt.subplot(1,4,i+1)
    plt.plot(t, acc[:,i])

plt.figure()
plt.title('Normed Records')
plt.subplot(1,3,1)
plt.plot(t, np.linalg.norm(acc, axis=1) )
plt.xlabel('acc')

plt.subplot(1,3,2)
plt.plot(t, np.linalg.norm(true_speed, axis=1) )
plt.xlabel('vel')

plt.subplot(1,3,3)
plt.plot(t, np.linalg.norm(true_force, axis=1) )
plt.xlabel('force')


plt.figure()





plt.show()


# print("speed 0 idx", idx_zero_speed)

# print("force at 0 speed", true_force[idx_zero_speed - 5])
# print("acc at 0 speed", acc[idx_zero_speed - 5])


# print("sevo poses", servo_poses[idx_zero_speed - 5])
# print("true poses", poses[idx_zero_speed - 5])

# print(np.max(np.sqrt(np.linalg.norm((servo_poses - poses[:, :2]), axis=-1))))
# argmax = np.argmax(np.linalg.norm((servo_poses - poses[:, :2]), axis=-1))
# print(np.argmax(np.linalg.norm((servo_poses - poses[:, :2]), axis=-1)))


# print(servo_poses[idx_zero_speed - 5] - poses[idx_zero_speed - 5, :2])