using System;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using SentimentAnalysisML.Model.DataModels;

namespace Sentiment_Analysis.Controllers
{
    public class SemanticController : Controller
    {
        [HttpGet]
        public IActionResult Analysis()
        {
            return View();
        }
        [HttpPost]
        public IActionResult Analysis(ModelInput input)
        {
            // Load the model
            MLContext mlContext = new MLContext();

            ITransformer mlModel = mlContext.Model.Load(@"..\Sentiment AnalysisML.Model\MLModel.zip", out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);
            //Input
            input.Year = DateTime.Now.Year;
            // Try model on sample data
            ModelOutput result = predEngine.Predict(input);
            ViewBag.Result = result;
            return View();
        }
    }
}
