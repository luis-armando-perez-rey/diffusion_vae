import json
import os
from modules.architectures.encoder_class import EncoderClass, EncoderClassMLP, EncoderClassMLPLabel, \
    EncoderClassMLPLatent, EncoderClassVGGLatent, EncoderClassVGGLabel
from modules.diffusion_vae_classes import Diffusion_Sphere_VAE
from modules.diffusion_vae_classes.parameters import diffusion_vae_parameters
from modules.diffusion_vae_classes.architectures import encoder_vgg, decoder_vgg
from modules.architectures.decoder_class import DecoderClass, DecoderClassMLPConditional, DecoderClassVGGConditional
from modules.diffusion_cvae_classes.conditional_latent_space import ConditionalLatentSpace, ConditionalCilinder
from modules.diffusion_cvae_classes.cvae import ConditionalVAEClass
from modules.diffusion_cvae_classes.cvae_fixed import define_cvae_vgg
from modules.diffusion_cvae_classes.cvae_fixed_semisupervised import define_cvae_vgg_semisupervised
from modules.diffusion_cvae_classes import cvae_fixed_semisupervised2, cvae_fixed_semisupervised3, cvae_fixed_semisupervised3b, cvae_fixed_semisupervised2b, cvae_fixed_semisupervised_balanced_beta
from modules.diffusion_cvae_classes import baseline_regression

def load_json(path):
    with open(path) as json_file:
        parameters = json.load(json_file)
    return parameters


def list_experiments(path):
    search_path = os.path.join(path, 'parameters')
    experiment_names = [name for name in os.listdir(search_path) if name[-1] == '_']
    return experiment_names

def load_delta_vae_parameters(experiment_path, num_experiment):
    vae_keyword_list = ["encoder", "decoder", "vae"]
    parameters = []
    for keyword in vae_keyword_list:
        parameters.append(load_parameters_keyword(experiment_path, keyword,num_experiment))
    return parameters


def load_encoder_class(experiment_path, num_experiment):
    # Load the parameters
    params = load_parameters_keyword(experiment_path, "encoder", num_experiment)
    # Remove the type entry from the dictionary
    params_no_type = {x: params[x] for x in params if x != "type"}
    # Create encoder class
    if params["type"] == "VGG":
        del params_no_type["intermediate_shape"]
        encoder_class = encoder_vgg.EncoderVGG(**params_no_type)
    else:
        print("Encoding type not recognized", params["type"])
        encoder_class = None
    return encoder_class

def load_decoder_class(experiment_path, num_experiment):
    # Load the parameters
    params = load_parameters_keyword(experiment_path, "decoder", num_experiment)
    # Remove the type entry from the dictionary
    params_no_type = {x: params[x] for x in params if x != "type"}
    # Create encoder class
    if params["type"] == "VGG":
        decoder_class = decoder_vgg.DecoderVGG(**params_no_type)
    else:
        print("Encoding type not recognized", params["type"])
        decoder_class = None
    return decoder_class

def load_delta_vae(experiment_path, num_experiment):
    # Load the parameters
    encoder_class = load_encoder_class(experiment_path,num_experiment)
    decoder_class = load_decoder_class(experiment_path, num_experiment)
    # Load the parameters
    params = load_parameters_keyword(experiment_path, "vae", num_experiment)
    if params["manifold"] == "hypersphere":
        del params["manifold"]
        diffusion_params = diffusion_vae_parameters.DiffusionVAEParams(**params)
        vae = Diffusion_Sphere_VAE.DiffusionSphereVAE(diffusion_params, encoder_class = encoder_class, decoder_class = decoder_class)
    return vae


    return decoder_class


def print_parameters_all_experiments(path):
    experiment_names = list_experiments(path)
    for num_experiment, experiment in enumerate(experiment_names):
        print("Num experiment {} Experiment {}".format(num_experiment, experiment))
        print("VAE {}".format(load_json(os.path.join(path, 'parameters', experiment, 'vae.json'))))
        print("Enc latent {}".format(load_json(os.path.join(path, 'parameters', experiment, 'enc_latent.json'))))
        print("Enc label {}".format(load_json(os.path.join(path, 'parameters', experiment, 'enc_label.json'))))
        print("Dec {}".format(load_json(os.path.join(path, 'parameters', experiment, 'dec.json'))))
        print("Latent {}".format(load_json(os.path.join(path, 'parameters', experiment, 'latent.json'))))


def load_parameters_fixed(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name+'.json'))
    return parameters

def load_parameters_keyword(path, keyword, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, keyword+"_"+experiment_name + '.json'))
    return parameters

def load_cvae_fixed(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= define_cvae_vgg(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters

def load_baseline_angles(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    baseline_model, _ = baseline_regression.define_baseline(**parameters)
    baseline_model.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return baseline_model, parameters

def load_cvae_fixed_semisupervised(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters

def load_cvae_fixed_semisupervised2(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= cvae_fixed_semisupervised2.define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters

def load_cvae_fixed_semisupervised2b(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= cvae_fixed_semisupervised2b.define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters


def load_cvae_fixed_semisupervised3(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= cvae_fixed_semisupervised3.define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters

def load_cvae_fixed_semisupervisedbeta(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= cvae_fixed_semisupervised_balanced_beta.define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters

def load_cvae_fixed_semisupervised3b(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    print('Experiment/timestamp to be loaded', experiment_name)
    parameters = load_json(os.path.join(path, 'parameters', experiment_name, experiment_name + '.json'))
    cvae, encoder_label, encoder_latent, decoder, _= cvae_fixed_semisupervised3b.define_cvae_vgg_semisupervised(**parameters)
    cvae.load_weights(os.path.join(path, 'weights_folder', experiment_name+'.h5'))
    return cvae, encoder_label, encoder_latent, decoder, parameters


def load_parameters_experiment(path, num_experiment):
    experiment_names = list_experiments(path)
    experiment_name = experiment_names[num_experiment]
    encoder_latent_parameters = load_json(os.path.join(path, 'parameters', experiment_name, 'enc_latent.json'))
    encoder_label_parameters = load_json(os.path.join(path, 'parameters', experiment_name, 'enc_label.json'))
    decoder_parameters = load_json(os.path.join(path, 'parameters', experiment_name, 'dec.json'))
    latent_parameters = load_json(os.path.join(path, 'parameters', experiment_name, 'latent.json'))
    vae_parameters = load_json(os.path.join(path, 'parameters', experiment_name, 'vae.json'))
    return encoder_latent_parameters, encoder_label_parameters, decoder_parameters, latent_parameters, vae_parameters


def latent_from_parameters(latent_parameters):
    dict_only_parameters = {x: latent_parameters[x] for x in latent_parameters if x != 'type'}
    if latent_parameters['type'] == "ConditionalLatentSpace":
        latent_space = ConditionalLatentSpace(**dict_only_parameters)
    elif latent_parameters['type'] == "ConditionalCilinder":
        latent_space = ConditionalCilinder(**dict_only_parameters)
    else:
        print("Unknown latent space")
        latent_space = None
    return latent_space


def encoder_label_define(latent, encoder_label_parameters):
    dict_only_parameters = {x: encoder_label_parameters[x] for x in encoder_label_parameters if x != 'type'}
    if encoder_label_parameters['type'] == "EncoderClassMLPLabel":
        encoder = EncoderClassMLPLabel(latent, **dict_only_parameters)
    elif encoder_label_parameters['type'] == "EncoderClassVGGLabel":
        encoder = EncoderClassVGGLabel(latent, **dict_only_parameters)
    else:
        encoder = None
    return encoder


def encoder_latent_define(latent, encoder_latent_parameters):
    dict_only_parameters = {x: encoder_latent_parameters[x] for x in encoder_latent_parameters if x != 'type'}
    if encoder_latent_parameters['type'] == "EncoderClassMLPLatent":
        encoder = EncoderClassMLPLatent(latent, **dict_only_parameters)
    elif encoder_latent_parameters['type'] == "EncoderClassVGGLatent":
        encoder = EncoderClassVGGLatent(latent, **dict_only_parameters)
    else:
        encoder = None
    return encoder


def decoder_define(latent, decoder_parameters):
    dict_only_parameters = {x: decoder_parameters[x] for x in decoder_parameters if x != 'type'}
    if decoder_parameters['type'] == "DecoderClass":
        decoder = DecoderClass(latent, **dict_only_parameters)
    elif decoder_parameters['type'] == "DecoderClassMLPConditional":
        decoder = DecoderClassMLPConditional(latent, **dict_only_parameters)
    elif decoder_parameters['type'] == "DecoderClassVGGConditional":
        decoder = DecoderClassVGGConditional(latent, **dict_only_parameters)
    else:
        return None
    return decoder


def load_experiment_from_path(path, num_experiment):
    experiment_name = list_experiments(path)[num_experiment]
    encoder_latent_parameters, encoder_label_parameters, decoder_parameters, latent_parameters, vae_parameters = load_parameters_experiment(
        path, num_experiment)
    dict_only_parameters = {x: vae_parameters[x] for x in vae_parameters if x != 'type'}
    latent = latent_from_parameters(latent_parameters)
    encoder_label = encoder_label_define(latent, encoder_label_parameters)
    encoder_latent = encoder_latent_define(latent, encoder_latent_parameters)
    decoder = decoder_define(latent, decoder_parameters)
    if vae_parameters['type']=='ConditionalVAEClass':
        cvae = ConditionalVAEClass(encoder_latent, encoder_label, decoder, latent)
    else:
        cvae = None
    weights_file = os.path.join(path,'weights_folder',experiment_name+'.h5')
    cvae.load_model_cvae_labelled(weights_file)

    return cvae

