import numpy as np
import os
import json

from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, RNN, Reshape, Permute, Dot, LSTM, Softmax
from keras.layers import LeakyReLU, UpSampling2D, Conv2DTranspose, Multiply, Activation,TimeDistributed, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2, l1
import keras.backend as K

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import model_from_json
import keras.backend as K

from dcase_models.model.container import KerasModelContainer

from .losses import prototype_loss
from .prototypes import Prototypes, DataInstances
from .layers import PrototypeLayer, WeightedSum


class APNet(KerasModelContainer):
    """
    Child class of ModelContainer with specific attributs and methods for
    APNet model.

    Attributes
    ----------
    prototypes : Prototypes
        Instance that includes prototypes information for visualization
        and debugging.
    data_instances : DataInstances
        Instance that includes data information for visualization
        and debugging.

    Methods
    -------
    get_prototypes(X_train, convert_audio_params=None, projection2D=None)
        Extract prototypes from model (embeddings, mel-spectrograms and audio
        if convert_audio_params is not None). Init self.prototypes instance. 

    get_data_instances(X_feat, X_train, Y_train, Files_names_train)
        Init self.data_instances object. Load data instances.
        
    debug_prototypes(self, X_train, force_get_prototypes=False)
        Function to debug the model by eliminating similar prototypes
    """

    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='softmax',
                 dilation_rate=(1,1), distance='euclidean',
                 use_weighted_sum=True, N_filters = [32,32,32],
                 **kwargs):
 
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn 
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn 
        self.pool_size_cnn = pool_size_cnn
        self.n_prototypes = n_prototypes 
        self.logits_activation = logits_activation
        self.dilation_rate = dilation_rate
        self.distance = distance 
        self.use_weighted_sum = use_weighted_sum
        self.N_filters = N_filters      

        self.prototypes = None
        self.data_instances = None

        super().__init__(model=model, model_path=model_path,
                        model_name='APNet', metrics=metrics, **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()
        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]    
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]  

        self.model_decoder = self.create_decoder(decoder_input_shape,mask1_shape,mask2_shape)

        x = Input(shape=(self.n_frames_cnn,self.n_freq_cnn), dtype='float32', name='input')

        feature_vectors, mask1, mask2 = self.model_encoder(x)
        deconv = self.model_decoder([feature_vectors,mask1,mask2])

        feature_vectors = Lambda(lambda x: x,name='features')(feature_vectors)

        prototypes_per_class = int(self.n_prototypes/self.n_classes)
        classes = []
        for j in range(self.n_classes):
            classes.append([j*prototypes_per_class, (j+1)*prototypes_per_class])
        prototype_distances = PrototypeLayer(self.n_prototypes, classes, distance=self.distance,
                                             name='prototype_distances', use_weighted_sum=self.use_weighted_sum)(feature_vectors)

        #prototype_similarity = Lambda(lambda x: K.log((x+1)/(x +K.epsilon())),name='similarity')(prototype_distances)#K.log((distance+1)/(distance+K.epsilon())) #
        #prototype_similarity = Lambda(lambda x: 1/(1+K.exp(x)),name='similarity')(prototype_distances)#K.log((distance+1)/(distance+K.epsilon())) #

        prototype_similarity_local = Lambda(lambda x: K.exp(-x),name='similarity_local')(prototype_distances)

        if self.distance == 'euclidean_patches':
            prototype_distances_sum = Lambda(lambda x: K.mean(x,2),name='prototype_distances_sum')(prototype_distances) #sum (25/01/2021)
            prototype_similarity_global = Lambda(lambda x: K.exp(-x),name='similarity_global')(prototype_distances_sum)
            prototype_similarity_local = Lambda(lambda x: K.max(x,(2,3)),name='maxpool_proto')(prototype_similarity_local)  
        else:
            prototype_distances_sum = Lambda(lambda x: x)(prototype_distances)

        if self.use_weighted_sum:
            if self.distance == 'euclidean_patches':
                mean = WeightedSum(name='mean')(prototype_similarity_global)
                #mean = WeightedSumDistances(name='sum0')([mean, prototype_similarity_local])
                mean = Lambda(lambda x: x[0]+x[1],name='sum0')([mean,prototype_similarity_local])

            else:
                mean = WeightedSum(name='mean')(prototype_similarity_local)
            logits = Dense(self.n_classes, use_bias=False, activation='linear', name='logits')(mean) #kernel_regularizer=l1(0.001) 
        else:
            logits = Dense(self.n_classes, use_bias=False, activation='linear', name='logits')(prototype_similarity_local)
                
        out = Activation(activation=self.logits_activation, name='out')(logits)
        
        #prototype_distances_sum = Lambda(lambda x: K.sum(x,-1),name='prototype_distances_sum_freq')(prototype_distances_sum)

        self.model = Model(inputs=x, outputs=[out, deconv, prototype_distances_sum])
                
        super().build()


    def create_encoder(self, activation = 'linear', name='encoder', use_batch_norm=False):
        #n_prototypes = prototypes_per_class * n_classes
        #SegNet: A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation

        x = Input(shape=(self.n_frames_cnn,self.n_freq_cnn), dtype='float32', name='input')
        y = Lambda(lambda x: K.expand_dims(x,-1),name='expand_dims')(x)
        y = Conv2D(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='conv1',dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu1')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool1')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling1')(y)
        bool_mask1 = Lambda(lambda t: K.greater_equal(t[0], t[1]),name='bool_mask1')([orig,y_up])
        mask1 = Lambda(lambda t: K.cast(t, dtype='float32'),name='mask1')(bool_mask1)

        y = Conv2D(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='conv2',dilation_rate=self.dilation_rate)(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu2')(y)
        orig = y
        y = MaxPooling2D(pool_size=self.pool_size_cnn, name='maxpool2')(y)
        y_up = UpSampling2D(size=self.pool_size_cnn, name='upsampling2')(y)
        bool_mask2 = Lambda(lambda t: K.greater_equal(t[0], t[1]),name='bool_mask2')([orig,y_up])
        mask2 = Lambda(lambda t: K.cast(t, dtype='float32'),name='mask2')(bool_mask2)    
        
        y = Conv2D(self.N_filters[2], self.filter_size_cnn, padding='same', activation=activation, name='conv3')(y)
        if use_batch_norm:
            y = BatchNormalization()(y) 
        y = LeakyReLU(name='leaky_relu3')(y)

        model = Model(inputs=x, outputs=[y, mask1, mask2], name=name)
        return model

    def create_decoder(self,input_shape,mask1_shape, mask2_shape,
                       activation = 'linear', final_activation='tanh',
                       name='decoder', use_batch_norm=False, N_filters_out=1):
    
        x = Input(shape=input_shape, dtype='float32', name='input')
        mask1 = Input(shape=mask1_shape, dtype='float32', name= 'input_mask1')
        mask2 = Input(shape=mask2_shape, dtype='float32', name= 'input_mask2')

        deconv = Conv2DTranspose(self.N_filters[1], self.filter_size_cnn, padding='same', activation=activation, name='deconv1')(x) 
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling2_1')(deconv)
        deconv = Multiply(name='multiply2')([mask2, deconv]) 
        #if use_batch_norm:
        #    deconv = BatchNormalization()(deconv)   
        deconv = LeakyReLU(name='leaky_relu4')(deconv)

        deconv = Conv2DTranspose(self.N_filters[0], self.filter_size_cnn, padding='same', activation=activation, name='deconv2')(deconv)     
        deconv = UpSampling2D(size=self.pool_size_cnn, name='upsampling3_1')(deconv)
        deconv = Multiply(name='multiply3')([mask1, deconv])    
    # if use_batch_norm:
    #     deconv = BatchNormalization()(deconv) 
        deconv = LeakyReLU(name='leaky_relu5')(deconv)

        deconv = Conv2DTranspose(N_filters_out, self.filter_size_cnn, padding='same', activation='linear', name='deconv3')(deconv)

        if N_filters_out == 1:
            deconv = Lambda(lambda x: K.squeeze(x, axis=-1),name='input_reconstructed')(deconv)

        if use_batch_norm:
            #deconv = LeakyReLU(name='leaky_relu6')(deconv)
            deconv = BatchNormalization()(deconv) 
        deconv = Activation(final_activation, name='final_activation')(deconv)

        model = Model(inputs=[x,mask1,mask2], outputs=deconv, name=name)        
        return model

    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False,
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
    #    n_classes = data_train[1].shape[1]
        n_classes = len(label_list)
        # define optimizer and compile
        losses = ['categorical_crossentropy'] + ['mean_squared_error'] + [prototype_loss] #+ ['mean_squared_error']  #

        # get number of prototypes and freq dimension of feature space
        features_shape = self.model.get_layer('features').output.get_shape().as_list()
        n_freqs_autoencoder = features_shape[2]
        prototypes_shape = self.model.get_layer('prototype_distances').output.get_shape().as_list()
        n_prototypes = prototypes_shape[1]
 #       n_instances_train = len(data_train[0])
 #       n_instances_val = len(data_val[0])

        # Init last layer
        if init_last_layer:   
            print('initialize last layer') 
            W = np.zeros((n_prototypes,n_classes))
            prototypes_per_class = int(n_prototypes / float(n_classes))
            print('prototypes_per_class', prototypes_per_class)
            for j in range(n_classes):
                W[prototypes_per_class*j:prototypes_per_class*(j+1),j] = 1/float(prototypes_per_class)#-1/float(prototypes_per_class)
                W[:prototypes_per_class*j,j] = -1/float(prototypes_per_class) #+1/float(prototypes_per_class)
                W[prototypes_per_class*(j+1):,j] = -1/float(prototypes_per_class)  #+1/float(prototypes_per_class)    
            W = W + 0.1*(np.random.rand(W.shape[0],W.shape[1]) - 0.5)
            self.model.get_layer('logits').set_weights([W]) 
            #self.model.get_layer('logits').trainable = False

        super().train(
            data_train, data_val,
            weights_path=weights_path, optimizer=optimizer,
            learning_rate=learning_rate, early_stopping=early_stopping,
            considered_improvement=considered_improvement,
            losses=losses, loss_weights=loss_weights,
            sequence_time_sec=sequence_time_sec,
            metric_resolution_sec=metric_resolution_sec,
            label_list=label_list, shuffle=shuffle,
            **kwargs_keras_fit
        )

    def model_input_to_embeddings(self):
        input_shape = self.model.layers[0].output_shape[1:] #model.get_layer('input').output_shape[1:]  #conv3    
        x = Input(shape=input_shape, dtype='float32')

        feature_vectors, mask1, mask2 = self.model.get_layer('encoder')(x)

        model = Model(inputs=x, outputs=[feature_vectors, mask1, mask2])

        return model     

    def model_input_to_distances(self, return_all=False):
        input_shape = self.model.layers[0].output_shape[1:] #model.get_layer('input').output_shape[1:]  #conv3    
        x = Input(shape=input_shape, dtype='float32')

        feature_vectors, mask1, mask2 = self.model.get_layer('encoder')(x)
        deconv = self.model.get_layer('decoder')([feature_vectors,mask1,mask2])
        features = self.model.get_layer('features')(feature_vectors)  
        prototype_distances = self.model.get_layer('prototype_distances')(feature_vectors) 

        prototype_similarity_local = self.model.get_layer('similarity_local')(prototype_distances)

        if self.distance == 'euclidean_patches':
            prototype_distances_sum = self.model.get_layer('prototype_distances_sum')(prototype_distances)
            prototype_similarity_global = self.model.get_layer('similarity_global')(prototype_distances_sum)
            prototype_similarity_local_max = self.model.get_layer('maxpool_proto')(prototype_similarity_local)
        else:
            prototype_distances_sum = Lambda(lambda x: x)(prototype_distances)

        if self.use_weighted_sum:
            if self.distance == 'euclidean_patches':
                prototype_similarity_global_mean = self.model.get_layer('mean')(prototype_similarity_global)
                mean = self.model.get_layer('sum0')([prototype_similarity_global_mean,prototype_similarity_local_max])
            else:
                mean = self.model.get_layer('mean')(prototype_similarity_local)
            logits = self.model.get_layer('logits')(mean)  
        else:
            logits = self.model.get_layer('logits')(prototype_similarity_local_max)        
        out = self.model.get_layer('out')(logits)

        if not return_all:
            model = Model(inputs=x, outputs=mean)
        else:
            model = Model(
                inputs=x,
                outputs=[
                    #prototype_similarity_global,
                    #prototype_similarity_global_mean,
                    mean,
                    prototype_similarity_local]
                    #prototype_similarity_local_max]
                )

        return model

    def model_embeddings_to_out(self):
        input_shape = self.model.get_layer('encoder').get_layer('conv3').output_shape[1:]  #conv3
        x = Input(shape=input_shape, dtype='float32')

        prototype_distances = self.model.get_layer('prototype_distances')(x)
        
        prototype_similarity_local = self.model.get_layer('similarity_local')(prototype_distances)

        if self.distance == 'euclidean_patches':
            prototype_distances_sum = self.model.get_layer('prototype_distances_sum')(prototype_distances)
            prototype_similarity_global = self.model.get_layer('similarity_global')(prototype_distances_sum)
            prototype_similarity_local = self.model.get_layer('maxpool_proto')(prototype_similarity_local)
        else:
            prototype_distances_sum = Lambda(lambda x: x)(prototype_distances)

        if self.use_weighted_sum:
            if self.distance == 'euclidean_patches':
                mean = self.model.get_layer('mean')(prototype_similarity_global)
                mean = self.model.get_layer('sum0')([mean,prototype_similarity_local])
            else:
                mean = self.model.get_layer('mean')(prototype_similarity_local)
            logits = self.model.get_layer('logits')(mean)  
        else:
            logits = self.model.get_layer('logits')(prototype_similarity_local)        
        out = self.model.get_layer('out')(logits)

        model = Model(inputs=x, outputs=[out,mean])     #prototype_distances
        return model   

                

    def model_embeddings_to_decoded(self):
        input_shape = self.model.get_layer('encoder').get_layer('conv3').output_shape[1:]    
        mask1_shape = self.model.get_layer('encoder').get_layer('mask1').input_shape[1:]
        mask2_shape = self.model.get_layer('encoder').get_layer('mask2').input_shape[1:]    
        x = Input(shape=input_shape, dtype='float32')
        mask1 = Input(shape=mask1_shape, dtype='float32')
        mask2 = Input(shape=mask2_shape, dtype='float32')
        
        decoded = self.model.get_layer('decoder')([x,mask1,mask2])

        model = Model(inputs=[x,mask1,mask2], outputs=decoded)

        return model  

    def model_with_new_prototypes(self, new_number_of_protos):                

        input_shape = self.model.layers[0].output_shape[1:]  #conv3

        x = Input(shape=input_shape, dtype='float32',name='input')
        feature_vectors, mask1, mask2 = self.model.get_layer('encoder')(x)
        feature_vectors = Lambda(lambda x: x,name='features')(feature_vectors)

        deconv = self.model.get_layer('decoder')([feature_vectors,mask1,mask2])
        
        prototype_distances = PrototypeLayer(new_number_of_protos,classes=None,distance=self.distance,
                                                name='prototype_distances', use_weighted_sum=True)(feature_vectors)

        prototype_similarity_local = Lambda(lambda x: K.exp(-x),name='similarity_local')(prototype_distances)

        n_classes = self.model.get_layer('out').output_shape[1]

        if self.distance == 'euclidean_patches':
            prototype_distances_sum = Lambda(lambda x: K.sum(x,2),name='prototype_distances_sum')(prototype_distances)
            prototype_similarity_global = Lambda(lambda x: K.exp(-x),name='similarity_global')(prototype_distances_sum)
            prototype_similarity_local = Lambda(lambda x: K.max(x,(2,3)),name='maxpool_proto')(prototype_similarity_local)  
        else:
            prototype_distances_sum = Lambda(lambda x: x)(prototype_distances)

        if self.use_weighted_sum:
            if self.distance == 'euclidean_patches':
                mean = WeightedSum(name='mean')(prototype_similarity_global)
                mean = Lambda(lambda x: x[0]+x[1],name='sum0')([mean,prototype_similarity_local])

            else:
                mean = WeightedSum(name='mean')(prototype_similarity_local)
            logits = Dense(n_classes, use_bias=False, activation='linear', name='logits')(mean) #kernel_regularizer=l1(0.001) 
        else:
            logits = Dense(n_classes, use_bias=False, activation='linear', name='logits')(prototype_similarity_local)
                
        out = Activation(activation=self.logits_activation, name='out')(logits)

        model = Model(inputs=x, outputs=[out,deconv,prototype_distances])

        return model

    def update_model_to_prototypes(self):
        new_number_of_protos = len(self.prototypes.embeddings)
        new_model = self.model_with_new_prototypes(new_number_of_protos)

        new_model.get_layer('mean').set_weights([self.prototypes.W_mean])
        new_model.get_layer('logits').set_weights([self.prototypes.W_dense])
        new_model.get_layer('prototype_distances').set_weights([self.prototypes.embeddings])

        self.model = new_model

    def model_without_weighted_sum(self):
        input_shape = self.model.layers[0].output_shape[1:] #model.get_layer('input').output_shape[1:]  #conv3    
        x = Input(shape=input_shape, dtype='float32')

        feature_vectors, mask1, mask2 = self.model.get_layer('encoder')(x)
        deconv = self.model.get_layer('decoder')([feature_vectors,mask1,mask2])
        features = self.model.get_layer('features')(feature_vectors)  
        prototype_distances = self.model.get_layer('prototype_distances')(feature_vectors) 

        prototype_similarity_local = self.model.get_layer('similarity_local')(prototype_distances)

        if self.distance == 'euclidean_patches':
            prototype_distances_sum = self.model.get_layer('prototype_distances_sum')(prototype_distances)
            prototype_similarity_global = self.model.get_layer('similarity_global')(prototype_distances_sum)
            prototype_similarity_local_max = self.model.get_layer('maxpool_proto')(prototype_similarity_local)
        else:
            prototype_distances_sum = Lambda(lambda x: x)(prototype_distances)

        if self.use_weighted_sum:
            if self.distance == 'euclidean_patches':
                prototype_similarity_global_mean = Lambda(lambda x: K.mean(x,-1),name='mean')(prototype_similarity_global)
                mean = self.model.get_layer('sum0')([prototype_similarity_global_mean,prototype_similarity_local_max])
            else:
                mean = Lambda(lambda x: K.mean(x,-1),name='mean')(prototype_similarity_local)
            logits = self.model.get_layer('logits')(mean)  
        else:
            logits = self.model.get_layer('logits')(prototype_similarity_local_max)        
        out = self.model.get_layer('out')(logits)

        if not return_all:
            model = Model(inputs=x, outputs=mean)
        else:
            model = Model(
                inputs=x,
                outputs=[
                    #prototype_similarity_global,
                    #prototype_similarity_global_mean,
                    mean,
                    prototype_similarity_local]
                    #prototype_similarity_local_max]
                )

        return model


    def get_prototypes(self, X_train, convert_audio_params=None, projection2D=None, random_masks=False):
        """
        Init self.prototypes object.

        Parameters
        ----------
        X_train : ndarray
            3D array with mel-spectrograms of train set.
            Shape = (N_instances, N_hops, N_mel_bands)
        convert_audio_params : dict, optional
            Dictionary with parameters needed for audio extraction
            keys: 'sr', 'scaler', 'mel_basis', 'audio_hop', 'audio_win'}
        projection2D : sklearn.decomposition, optional
            sklean Object to project the embedding space into a 2D space.
            For instance, use PCA class. 
        """
        self.prototypes = Prototypes(self, X_train, projection2D=projection2D, convert_audio_params=convert_audio_params, random_masks=random_masks)
    
    def get_data_instances(self, X_feat, X_train, Y_train, Files_names_train, projection2D=None):
        """
        Init self.instances object.

        Parameters
        ----------
        X_feat : ndarray
            4D array with the embeddings of trainig data.
            Shape (N_files, N_hops_feat, N_freqs_feat, N_filters)
            Note that we use just one mel-spectrogram per file.
        X_train : ndarray
            3D array with mel-spectrograms of train set.
            Shape = (N_files, N_hops, N_mel_bands)
        Y_train : ndarray
            2D array with the annotations of train set (one hot encoding).
            Shape (N_files, N_classes)
        Files_names_train : list
            List of paths to audio files. Len of this list has to be N_files
        """
        self.data_instances = DataInstances(
            X_feat, X_train, Y_train, Files_names_train, n_classes=Y_train.shape[1],use_kmeans=False, projection2D=projection2D)


    def refine_prototypes(self, X_train, gamma=2, force_get_prototypes=False): #gamma=2
        """
        Refine the model by deleting similar prototypes.

        Parameters
        ----------
        X_train : ndarray
            3D array with mel-spectrograms of train set.
            Shape = (N_files, N_hops, N_mel_bands)
        gamma : float
            Threshold of function decision when deleting prototypes
        force_get_prototypes : bool
            If True, self.get_prototypes() function is called
        """
        if (self.prototypes is None) | (force_get_prototypes == True):
            self.get_prototypes(X_train)
        print(self.prototypes.classes)
        self.prototypes.sort()
        print(self.prototypes.classes)
        #prototypes_embeddings,prototypes_melspec,_,prototypes_classes, prototypes_distances,_ = self.prototypes.get_all_instances()
        N_protos = self.prototypes.get_number_of_instances()
        
        prototypes_distances = np.power(np.expand_dims(self.prototypes.embeddings, 0) - np.expand_dims(self.prototypes.embeddings, 1), 2)
        print(prototypes_distances.shape)
        prototypes_distances = np.mean(prototypes_distances, axis=(2,3,4))
        print(prototypes_distances.shape)
        #prototypes_distances = np.power(self.prototypes.embeddings[prototype,:,:,j] - 
        #                             self.prototypes.embeddings[prototype,:,:,k],2))
        
        prototypes_distances_sum = np.mean(prototypes_distances, axis=0)

        #gamma = np.mean(prototypes_distances_sum) - 5*np.std(prototypes_distances_sum)
        
        print(prototypes_distances_sum)
        print(gamma)

        prototypes_distances_original = prototypes_distances.copy()

        prototypes_distances[prototypes_distances==0] = 8000

        gamma = np.min(prototypes_distances) + np.mean(prototypes_distances_sum)/2

        indexes = np.arange(N_protos)
        deletes = []
        keep = []
        for j in range(N_protos):
            if np.sum(deletes==j)>0:
                continue
            class_ix = self.prototypes.classes[j]

            #prototypes_distances_class = prototypes_distances_original[self.prototypes.classes==class_ix, self.prototypes.classes==class_ix]
            #print(prototypes_distances_class.shape)

            #gamma = np.mean(prototypes_distances_class) - 1.5*np.std(prototypes_distances_class)

            min_distance = np.amin(prototypes_distances[j])
            argmin = np.argmin(prototypes_distances[j])
            print(min_distance, argmin, gamma)
            if (min_distance < gamma) & (j not in keep) & (self.prototypes.classes[argmin] == class_ix):
                deletes.append(j)
                keep.append(argmin)

            # prototypes_distances_j = prototypes_distances_sum[j]
            # delete = np.where((self.prototypes.classes==class_ix) & (indexes> j) & 
            #                   (np.abs(prototypes_distances_sum - prototypes_distances_j)<gamma)) #<
            # if len(delete[0])>0:
            #     keep[j] = len(delete[0])+1
            #     if deletes is None:
            #         deletes = delete[0]
            #     else:
            #         deletes = np.concatenate((deletes,delete[0]))

        deletes = np.unique(deletes)
        print(deletes)
        
        self.prototypes.remove_instance(deletes)

        print(self.prototypes.classes)

        new_prototypes_distances_sum  = np.delete(prototypes_distances_sum, deletes)
        
        W_dense_new,W_mean_new = self.prototypes.get_weights()

        prototypes_feat_new,prototypes_mel_new,_,prototypes_classes_new, _ = self.prototypes.get_all_instances()
        
        N_protos_new = self.prototypes.get_number_of_instances()

        # redefine self.model with the debugged model
        #self.model = debugg_model(self.model, N_protos_new)

        self.model = self.model_with_new_prototypes(N_protos_new)
        self.model.get_layer('prototype_distances').set_weights([prototypes_feat_new])
        self.model.get_layer('mean').set_weights([W_mean_new])
        self.model.get_layer('logits').set_weights([W_dense_new])

        print('%d prototypes were deleted' % (N_protos - N_protos_new))

    def refine_channels(self, X_train, force_get_prototypes=False, distance='euclidean'):
        if (self.prototypes is None) | (force_get_prototypes == True):
            self.get_prototypes(X_train)

        conv3, bias_conv3 = self.model.get_layer('encoder').get_layer('conv3').get_weights()
        n_channels = len(bias_conv3)
        d = np.zeros((n_channels,n_channels))

        for prototype in range(len(self.prototypes.embeddings)):
            for j in range(n_channels):
                for k in range(n_channels): 
                    d[j,k] += np.sum(np.power(self.prototypes.embeddings[prototype,:,:,j] - 
                                     self.prototypes.embeddings[prototype,:,:,k],2))

        d_orig = d.copy()
        top_N = 8
        top_N_orig = top_N
        np.fill_diagonal(d, 1000)
        d_b = np.tril(d)
        
        d_b[d_b==0] = 1000
        
        stop = False
        while(stop==False):
            y = [x for x in d_orig.flatten() if x != 0]
            #print(y)
            idx = np.argpartition(-d_b, d_b.size - top_N, axis=None)[-top_N:]
            result = np.column_stack(np.unravel_index(idx, d_b.shape))
            # gamma = np.mean(y)-1.5*np.std(y) #np.amax(d_orig)/10.
            # result = np.argwhere(d_b < gamma)
            # print(gamma, np.amin(d_b))
            # if len(result) == 0:
            #     gamma = np.mean(y)-1.0*np.std(y)
            #     result = np.argwhere(d_b < gamma)
            delete = []
            delete.append(result[0,0])
            for j in range(1,len(result)):
                if result[j,0] in delete:
                    delete.append(result[j,1])
                else:
                    delete.append(result[j,0])
            delete = np.asarray(delete)
            delete = np.unique(delete)
            print('delete %d channels'%len(delete))
           # stop = True
            if len(delete)==top_N_orig:
               stop = True
            else:
               top_N +=1
        print(delete)

        new_prototypes = np.delete(self.prototypes.embeddings, delete, 3)
        new_conv3 = np.delete(conv3, delete, 3)
        new_bias_conv3 = np.delete(bias_conv3, delete, 0)
        
        input_shape = self.model.get_layer('input').output_shape[1:]
        output_shape = self.model.get_layer('out').output_shape[1:]
        print('input_shape', input_shape)
        print('output_shape', output_shape)
        n_prototypes = self.model.get_layer('prototype_distances').get_weights()[0].shape[0]

        new_N_filters = n_channels - len(delete)
        self.N_filters = [self.N_filters[0], self.N_filters[1], new_N_filters]
        self.distance = distance
        self.n_prototypes = len(new_prototypes)
        self.n_frames_cnn = input_shape[0] 
        self.n_freq_cnn = input_shape[1]
        self.n_classes = output_shape[-1]

        # get old weights
        old_weights = {}
        for layer in ['mean', 'logits']:
            old_weights[layer] = self.model.get_layer(layer).get_weights()
        for layer in ['conv1', 'conv2']:
            old_weights[layer] = self.model.get_layer('encoder').get_layer(layer).get_weights()

        # re-create self.model
        self.build()

        self.model.get_layer('encoder').summary()

        # set old weights
        for layer in ['mean', 'logits']:
            print(layer)
            self.model.get_layer(layer).set_weights(old_weights[layer])

        for layer in ['conv1', 'conv2']:
            self.model.get_layer('encoder').get_layer(layer).set_weights(old_weights[layer])    

        self.model.get_layer('prototype_distances').set_weights([new_prototypes])
        self.model.get_layer('encoder').get_layer('conv3').set_weights([new_conv3,new_bias_conv3])


class APNetFewShot(APNet):
    """
    Child class of ModelContainer with specific attributs and methods for
    APNet model.

    Attributes
    ----------
    prototypes : Prototypes
        Instance that includes prototypes information for visualization
        and debugging.
    data_instances : DataInstances
        Instance that includes data information for visualization
        and debugging.

    Methods
    -------
    get_prototypes(X_train, convert_audio_params=None, projection2D=None)
        Extract prototypes from model (embeddings, mel-spectrograms and audio
        if convert_audio_params is not None). Init self.prototypes instance. 

    get_data_instances(X_feat, X_train, Y_train, Files_names_train)
        Init self.data_instances object. Load data instances.
        
    debug_prototypes(self, X_train, force_get_prototypes=False)
        Function to debug the model by eliminating similar prototypes
    """

    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64, n_freq_cnn=128,
                 filter_size_cnn=(5, 5), pool_size_cnn=(2,2),
                 n_prototypes=50, logits_activation='softmax',
                 dilation_rate=(1,1), distance='euclidean',
                 use_weighted_sum=True, N_filters = [32,32,32],
                 **kwargs):


        super().__init__(model=model, model_path=model_path, metrics=metrics,
                 n_classes=n_classes, n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn,
                 filter_size_cnn=filter_size_cnn, pool_size_cnn=pool_size_cnn,
                 n_prototypes=n_prototypes, logits_activation=logits_activation,
                 dilation_rate=dilation_rate, distance=distance,
                 use_weighted_sum=use_weighted_sum, N_filters = N_filters,
                 **kwargs)


    def build(self):
        self.model_encoder = self.create_encoder()
        decoder_input_shape = self.model_encoder.get_layer('conv3').output_shape[1:]    
        mask1_shape = self.model_encoder.get_layer('mask1').input_shape[1:]
        mask2_shape = self.model_encoder.get_layer('mask2').input_shape[1:]  

        self.model_decoder = self.create_decoder(decoder_input_shape,mask1_shape,mask2_shape)

        x = Input(shape=(self.n_frames_cnn,self.n_freq_cnn), dtype='float32', name='input')

        feature_vectors, mask1, mask2 = self.model_encoder(x)
        deconv = self.model_decoder([feature_vectors,mask1,mask2])

        feature_vectors = Lambda(lambda x: x,name='features')(feature_vectors)

        prototypes_per_class = int(self.n_prototypes/self.n_classes)
        classes = []
        for j in range(self.n_classes):
            classes.append([j*prototypes_per_class, (j+1)*prototypes_per_class])
        prototype_layer = PrototypeLayer(self.n_prototypes, classes, distance=self.distance,
                                             name='prototype_distances', use_weighted_sum=self.use_weighted_sum)
        prototype_distances = prototype_layer(feature_vectors)

        similarity_fn = Lambda(lambda x: K.exp(-x), name='similarity_local') 
        prototype_similarity = similarity_fn(prototype_distances)

        if self.use_weighted_sum:
            weightedsum = WeightedSum(name='mean')
            mean = weightedsum(prototype_similarity)
            logits = Dense(self.n_classes, use_bias=False, activation='linear', name='logits')(mean) 
        else:
            logits = Dense(self.n_classes, use_bias=False, activation='linear', name='logits')(prototype_similarity)
                
        out = Activation(activation=self.logits_activation, name='out')(logits)
        
        #prototype_distances_sum = Lambda(lambda x: K.sum(x,-1),name='prototype_distances_sum_freq')(prototype_distances_sum)

        x_fs = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input_fs')
        x_fs_neg = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input_fs_neg')
        x_Q = Input(shape=(self.n_frames_cnn, self.n_freq_cnn), dtype='float32', name='input_Q')

        features_fs, _, _ = self.model_encoder(x_fs)
        distances_fs = prototype_layer(features_fs)
        similarity_fs = similarity_fn(distances_fs)
        if self.use_weighted_sum:
            similarity_fs = weightedsum(similarity_fs)
        mean_similarity_fs = Lambda(lambda x: K.mean(x, axis=0, keepdims=True), name='mean_similarity_fs')(similarity_fs)

        features_fs_neg, _, _ = self.model_encoder(x_fs_neg)
        distances_fs_neg = prototype_layer(features_fs_neg)
        similarity_fs_neg = similarity_fn(distances_fs_neg)
        if self.use_weighted_sum:
            similarity_fs_neg = weightedsum(similarity_fs_neg)
        mean_similarity_fs_neg = Lambda(lambda x: K.mean(x, axis=0, keepdims=True), name='mean_similarity_fs_neg')(similarity_fs_neg)

        features_Q, _, _ = self.model_encoder(x_Q)
        distances_Q = prototype_layer(features_Q)
        similarity_Q = similarity_fn(distances_Q)
        if self.use_weighted_sum:
            similarity_Q = weightedsum(similarity_Q)

        #decision_fn = Lambda(lambda x: K.sum(K.pow(x[0] - x[1], 2), axis=-1, keepdims=True), name='decision_fn')([similarity_Q, mean_similarity_fs])
        #decision_fn = Lambda(lambda x: K.pow(x[0] - x[1], 2), name='decision_fn')([similarity_Q, mean_similarity_fs])
        #decision_fn = Lambda(lambda x: x[0] - x[1], name='decision_fn')([similarity_Q, mean_similarity_fs])
        #decision_fn = Lambda(lambda x: K.pow(x[0] - x[1], 2) - K.pow(x[0] - x[2], 2), name='decision_fn')([similarity_Q, mean_similarity_fs, mean_similarity_fs_neg])

        decision_fn = Lambda(lambda x: K.exp(-K.sum(K.pow(x[0] - x[1], 2), axis=-1, keepdims=True)), name='decision_fn')([similarity_Q, mean_similarity_fs])
        decision_fn_neg = Lambda(lambda x: K.exp(-K.sum(K.pow(x[0] - x[1], 2), axis=-1, keepdims=True)), name='decision_fn_neg')([similarity_Q, mean_similarity_fs_neg])

        decision_fn = Concatenate()([decision_fn_neg, decision_fn])

        out_Q = Activation(activation='sigmoid', name='out_Q')(decision_fn)

        #out_Q = Dense(2, activation='softmax')(decision_fn)
        #decision_fn = Lambda(lambda x: x - 0.5, name='decision_fn1')(decision_fn)#
        #decision_fn = BatchNormalization()(decision_fn)
        #out_Q = Activation(activation='sigmoid', name='out_Q')(decision_fn)

        self.model = Model(inputs=[x, x_fs, x_fs_neg, x_Q], outputs=[out, deconv, prototype_distances, out_Q])
                
    def train(self, data_train, data_val, weights_path='./',
              optimizer='Adam', learning_rate=0.001, early_stopping=100,
              considered_improvement=0.01,
              loss_weights=[10,5,5,5], sequence_time_sec=0.5,
              metric_resolution_sec=1.0, label_list=[],
              shuffle=True, init_last_layer=False,
              **kwargs_keras_fit):

        """
        Specific training function for APNet model
        """
    #    n_classes = data_train[1].shape[1]
        n_classes = len(label_list)
        # define optimizer and compile
        losses = ['categorical_crossentropy'] + ['mean_squared_error'] + [prototype_loss] + ['categorical_crossentropy']  #

        # get number of prototypes and freq dimension of feature space
        features_shape = self.model.get_layer('features').output.get_shape().as_list()
        n_freqs_autoencoder = features_shape[2]
        prototypes_shape = self.model.get_layer('mean_similarity_fs').output.get_shape().as_list()
        n_prototypes = prototypes_shape[1]
 #       n_instances_train = len(data_train[0])
 #       n_instances_val = len(data_val[0])

        # Init last layer
        if init_last_layer:   
            print('initialize last layer') 
            W = np.zeros((n_prototypes,n_classes))
            prototypes_per_class = int(n_prototypes / float(n_classes))
            print('prototypes_per_class', prototypes_per_class)
            for j in range(n_classes):
                W[prototypes_per_class*j:prototypes_per_class*(j+1),j] = 1/float(prototypes_per_class)#-1/float(prototypes_per_class)
                W[:prototypes_per_class*j,j] = -1/float(prototypes_per_class) #+1/float(prototypes_per_class)
                W[prototypes_per_class*(j+1):,j] = -1/float(prototypes_per_class)  #+1/float(prototypes_per_class)    
            W = W + 0.1*(np.random.rand(W.shape[0],W.shape[1]) - 0.5)
            self.model.get_layer('logits').set_weights([W]) 
            #self.model.get_layer('logits').trainable = False

        import keras.optimizers as optimizers
        optimizer_function = getattr(optimizers, optimizer)
        opt = optimizer_function(lr=learning_rate)

        self.model.compile(loss=losses, optimizer=opt,
                           loss_weights=loss_weights)

        file_weights = os.path.join(weights_path, 'best_weights.hdf5')
        file_log = os.path.join(weights_path, 'training.log')
        
        kwargs_keras_fit.pop('buffer_size')
        print(kwargs_keras_fit)
        self.model.fit(
            x=data_train[0], y=data_train[1], shuffle=False,
            #callbacks=[metrics_callback, log],
            #validation_data=validation_data,
            **kwargs_keras_fit
        )
        self.model.save_weights(file_weights)

        # if self.metrics[0] == 'classification':
        #     metrics_callback = ClassificationCallback(
        #         data_val, file_weights=file_weights,
        #         early_stopping=early_stopping,
        #         considered_improvement=considered_improvement,
        #         label_list=label_list
        #     )
        # elif self.metrics[0] == 'sed':
        #     metrics_callback = SEDCallback(
        #         data_val, file_weights=file_weights,
        #         early_stopping=early_stopping,
        #         considered_improvement=considered_improvement,
        #         sequence_time_sec=sequence_time_sec,
        #         metric_resolution_sec=metric_resolution_sec,
        #         label_list=label_list
        #     )
        # elif self.metrics[0] == 'tagging':
        #     metrics_callback = TaggingCallback(
        #         data_val, file_weights=file_weights,
        #         early_stopping=early_stopping,
        #         considered_improvement=considered_improvement,
        #         label_list=label_list
        #     )
        # else:
        #     metrics_callback = ModelCheckpoint(
        #         filepath=file_weights,
        #         save_weights_only=True,
        #         monitor='val_loss',
        #         mode='min',
        #         save_best_only=True,
        #         verbose=1)

        # log = CSVLogger(file_log)
        # kwargs_keras_fit.pop('buffer_size')
        # validation_data = None
        # if metrics_callback.__class__ is ModelCheckpoint:
        #     if data_val.__class__ is DataGenerator:
        #         validation_data = KerasDataGenerator(data_val)
        #     else:
        #         validation_data = data_val
        # if type(data_train) in [list, tuple]:
        #     self.model.fit(
        #         x=data_train[0], y=data_train[1], shuffle=shuffle,
        #         callbacks=[metrics_callback, log],
        #         validation_data=validation_data,
        #         **kwargs_keras_fit
        #     )
        # else:
        #     if data_train.__class__ is DataGenerator:
        #         data_train = KerasDataGenerator(data_train)
        #     kwargs_keras_fit.pop('batch_size')
            
        #     self.model.fit_generator(
        #         generator=data_train,
        #         callbacks=[metrics_callback, log],
        #         validation_data=validation_data,
        #         shuffle=False,
        #         **kwargs_keras_fit
        #         # use_multiprocessing=True,
        #         # workers=6)
        #     )

class AttRNNSpeechModel(KerasModelContainer):
     # https://github.com/douglas125/SpeechCmdRecognition/blob/master/SpeechModels.py

    def __init__(self, model=None, model_path=None, metrics=['classification'],
                 n_classes=10, n_frames_cnn=64,
                 n_freq_cnn=128, filter_size_cnn=(5, 5), pool_size_cnn=(2, 2),
                 n_dense_cnn=64, n_channels=0):
        """ Initialization of the SB-CNN model.

        """
        self.n_classes = n_classes
        self.n_frames_cnn = n_frames_cnn
        self.n_freq_cnn = n_freq_cnn
        self.filter_size_cnn = filter_size_cnn
        self.pool_size_cnn = pool_size_cnn
        self.n_dense_cnn = n_dense_cnn
        self.n_channels = n_channels

        super().__init__(
            model=model, model_path=model_path,
            model_name='SB_CNN', metrics=metrics
        )

    def build(self):

        inputs = Input(shape=(self.n_frames_cnn, self.n_freq_cnn),
                    dtype='float32', name='input')
        x = Lambda(lambda x: K.expand_dims(x, -1), name='lambda')(inputs)

        x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        # x = Reshape((125, 80)) (x)
        # keras.backend.squeeze(x, axis)
        x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

        x = Bidirectional(LSTM(64, return_sequences=True)
                            )(x)  # [b_s, seq_len, vec_dim]
        x = Bidirectional(LSTM(64, return_sequences=True)
                            )(x)  # [b_s, seq_len, vec_dim]

        xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
        query = Dense(128)(xFirst)  # [b_s, vec_dim]

        # dot product attention
        attScores = Dot(axes=[1, 2])([query, x]) # [b_s, seq_len]
        attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

        # rescale sequence
        attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

        x = Dense(64, activation='relu')(attVector) # [b_s, 64]
        x = Dense(32)(x) # [b_s, 32]

        output = Dense(self.n_classes, activation='softmax', name='output')(x)  # [b_s, n_classes]

        self.model = Model(inputs=[inputs], outputs=[output])