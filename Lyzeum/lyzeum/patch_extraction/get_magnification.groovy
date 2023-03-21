/**
 * Script to print the magnification of an image.
 */
 import java.net.URI

String WSIPath = args[0]

def uri = new URI(WSIPath)
def server = new qupath.lib.images.servers.bioformats.BioFormatsServerBuilder().buildServer(uri)
def imageData = new ImageData(server)

def metadata = server.getMetadata()


String start_pattern = "slide magnification<<<"
String end_pattern = ">>>slide magnification"

print start_pattern + metadata.getMagnification().toString() + end_pattern
