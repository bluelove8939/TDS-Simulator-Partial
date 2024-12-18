import copy
import torch
import torch.quantization.quantize_fx as quantize_fx

from tds_sim.models.tools.progressbar import ProgressBar


class QuantizationModule(object):
    def __init__(self, tuning_dataloader=None, criterion=None, optimizer=None):
        self.tuning_dataloader = tuning_dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def quantize(self, model, example_inputs, default_qconfig='fbgemm', citer=0, verbose=2):
        device = 'cpu'
        quant_model = copy.deepcopy(model)
        quant_model.eval()
        qconfig = torch.quantization.get_default_qconfig(default_qconfig)
        qconfig_dict = {"": qconfig}

        if verbose:
            print("\nQuantization Configs")
            print(f"- criterion: {self.criterion}")
            print(f"- qconfig: {default_qconfig}")
            print(f"- device:  {device}")

        if verbose: print("preparing quantization (symbolic tracing)")
        model_prepared = quantize_fx.prepare_fx(quant_model, qconfig_dict, example_inputs)
        if verbose == 2: print(model_prepared)

        if citer: self.calibrate(model_prepared, citer=citer, verbose=verbose)
        model_quantized = quantize_fx.convert_fx(model_prepared)
        return model_quantized

    def calibrate(self, model, citer, verbose=2):
        if self.tuning_dataloader is None or self.criterion is None or self.optimizer is None:
            raise Exception('Error!')

        device = 'cpu'
        maxiter = min(len(self.tuning_dataloader), citer)
        if verbose == 1:
            print(f'\rcalibration iter: {0:3d}/{maxiter:3d}', end='')
        elif verbose:
            print(f'calibration iter: {0:3d}/{maxiter:3d}')

        cnt = 1
        model.eval()                                      # set to evaluation mode

        with torch.no_grad():                             # do not save gradient when evaluation mode
            for image, target in self.tuning_dataloader:  # extract input and output data
                image = image.to(device)
                model(image)                              # forward propagation
                if verbose == 1:
                    print(f'\rcalibration iter: {cnt:3d}/{maxiter:3d}', end='')
                elif verbose:
                    print(f'calibration iter: {cnt:3d}/{maxiter:3d}')

                cnt += 1
                if cnt > citer: break

        if verbose == 1: print('\n')
        elif verbose: print()
