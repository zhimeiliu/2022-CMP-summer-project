/**
 * Script to export tiles from specified regions on whole slide images
 */
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.ImageRegion
import java.net.URI

// Organise the command-line arguments
String WSIPath = args[0]
double downsample = args[1].toDouble()
int tileSize = args[2].toInteger()
int overlap = args[3].toInteger()
String outDir = args[4]
String tileFormat = args[5]
int x = args[6].toInteger()
int y = args[7].toInteger()
int width = args[8].toInteger()
int height = args[9].toInteger()
String task = args[10]


def uri = new URI(WSIPath)
def server = new qupath.lib.images.servers.bioformats.BioFormatsServerBuilder().buildServer(uri)
def imageData = new ImageData(server)


if (x == 0 & y == 0 && width == 0 && height == 0){
    width = server.getWidth()
    height = server.getHeight()
}


def region = RegionRequest.createInstance(imageData.getServerPath(), downsample, x, y, width, height)


def pathOutput = buildFilePath(outDir)
mkdirs(pathOutput)


if (task == "generate_patches"){
    new TileExporter(imageData)
        .region(region)
        .downsample(downsample)
        .imageExtension(tileFormat)
        .tileSize(tileSize)
        .annotatedTilesOnly(false)
        .overlap(overlap)
        .includePartialTiles(false)
        .writeTiles(pathOutput)
    }

if (task == "generate_overview"){
    writeImageRegion(server, region, pathOutput)
}

