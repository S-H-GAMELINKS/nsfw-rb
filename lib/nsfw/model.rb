require "onnxruntime"

module NSFW
  class Model
    ROOT_PATH        = Pathname.new(__dir__).parent.parent
    MODEL_PATH       = "#{ROOT_PATH}/vendor/onnx_models/nsfw.onnx"
    CATEGORIES       = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    SAFETY_THRESHOLD = 0.7

    attr_reader :model

    def initialize(lazy: false)
      load_model! unless lazy
    end

    def predict(tensor)
      prediction = make_prediction(tensor)
      format_prediction(prediction)
    end

    CATEGORIES.each do |category|
      define_method "#{category}?" do
        puts @prediction.inspect
        @prediction[category] >= SAFETY_THRESHOLD
      end
    end

    def safe?(image)
      predict(image.tensor)
      neutral? || drawings?
    end

    def unsafe?(image)
      predict(image.tensor)
      porn? || hentai? || sexy?
    end

    def loaded?
      !@model.nil?
    end

    def reshape_tensor(tensor)
      OnnxRuntime::Utils.reshape(tensor, [1, 224, 224, 3])
    end

    private

    def load_model!
      @model ||= OnnxRuntime::Model.new(MODEL_PATH)
    end

    def make_prediction(tensor)
      load_model! unless loaded?
      @model.predict({ "input" => reshape_tensor(tensor) })
    end

    def format_prediction(prediction)
      results = prediction.fetch("prediction").first
      @prediction = CATEGORIES.zip(results).sort{|a,b| b.last - a.last }.to_h
    end
  end
end