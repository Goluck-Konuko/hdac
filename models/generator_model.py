import torch
import torch.nn as nn
from typing import Dict, Any
from .components.utils import Vgg19, ImagePyramide, Transform

def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """
    def __init__(self, kp_extractor:Any = None, generator: Any =None, discriminator: Any=None, config: Dict[str, Any]=None):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.train_params = config['train_params']
        self.scales = self.train_params['scales']
        if discriminator is not None:
            self.disc_scales = self.discriminator.scales
        else:
            self.disc_scales = [1]
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)

        self.loss_weights = self.train_params['loss_weights']
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()

    def forward(self, x: Dict[str,torch.tensor]) -> tuple:
        kp_driving = self.kp_extractor(x['driving'])
        kp_source = self.kp_extractor(x['source'])
        
        if self.config['dataset_params']['base_layer']:
            generated = self.generator(x['source'],base_layer=x['hevc'],kp_source= kp_source,kp_driving =kp_driving)
        else:
            generated = self.generator(x['source'],kp_source= kp_source,kp_driving =kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
    
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    if not torch.isnan(value):
                        value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0 and self.discriminator !=None:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                if not torch.isnan(value):
                    value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        if not torch.isnan(value) and not torch.isinf(value):
                            value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0 and self.kp_extractor != None:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        return (loss_values, generated)