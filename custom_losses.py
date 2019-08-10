import tensorflow as tf
from keras.losses import binary_crossentropy

def combined_loss(generated,beta,power):
	def generator_loss(y_true,y_generated):
		#gradient descent on alpha value
		encirclement_loss = tf.reduce_mean(binary_crossentropy(y_true, y_generated))
		center = tf.reduce_mean(generated, axis=0, keepdims=True)
		distance_xy = tf.pow(tf.abs(tf.subtract(generated,center)),power)
		distance = tf.reduce_sum(distance_xy, (1,2,3))
		avg_distance = tf.reduce_mean(tf.pow(distance, 1/power))
		dispersion_loss = tf.reciprocal(avg_distance)
		
		loss = encirclement_loss + beta*dispersion_loss
		return loss
	return generator_loss