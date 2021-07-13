from WCT2.transfer import StyleTransfer
from config import StyleTransferConfig

style_config = StyleTransferConfig
style_trans = StyleTransfer(style_config)
style_trans.whole_dir_transfer(
    cont_dir=style_config.content_dir, 
    sty_dir=style_config.style_dir,
    out_dir=style_config.output_dir)